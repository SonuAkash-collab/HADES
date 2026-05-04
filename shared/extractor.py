from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Iterable

from transformers import pipeline

from .triple import KnowledgeTriple




@lru_cache(maxsize=1)
def _load_rebel_model():
    """Loads the REBEL seq2seq transformer model and tokenizer manually."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model_name = "Babelscape/rebel-large"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer





def _sanitize_pdf_text(text: str) -> str:
    """Generic sanitization for PDF-derived text artifacts."""
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    # Fix line-break hyphenation (e.g., 'knowl-\n edge' -> 'knowledge')
    text = re.sub(r'([a-zA-Z])-\s*\n\s*([a-zA-Z])', r'\1\2', text)
    # Collapse multiple spaces/newlines into a single space
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _chunk_text(text: str, sentences_per_chunk: int = 3) -> list[str]:
    """Splits text into chunks of ~3-4 sentences to fit REBEL's token limit."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _parse_rebel_output(text: str) -> list[dict]:
    """
    Standard REBEL decoding logic to parse <triplet>, <subj>, and <obj> tokens.
    Mapping: <triplet> head <subj> tail <obj> type
    """
    triplets = []
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current = 'x'
    
    # Process tokens into head, tail, and type dictionaries
    # Strip <pad> tokens that REBEL emits — these corrupt verb labels downstream
    for token in text.replace("<s>", "").replace("</s>", "").replace("<pad>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                relation = ''
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
                
    if relation != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
        
    return triplets


def _extract_rebel_triples(text: str) -> list[KnowledgeTriple]:
    """
    Performs inference using REBEL and maps results to KnowledgeTriple objects.
    """
    model, tokenizer = _load_rebel_model()
    chunks = _chunk_text(text)
    all_triples: list[KnowledgeTriple] = []
    
    for chunk in chunks:
        # Prepare inputs
        inputs = tokenizer(chunk, return_tensors="pt")
        
        # Standard REBEL inference parameters
        gen_outputs = model.generate(
            inputs["input_ids"],
            max_length=256, 
            length_penalty=0, 
            num_beams=3, 
            num_return_sequences=3,
        )
        
        for output in gen_outputs:
            # Decode tensor output to text with special tokens
            decoded_text = tokenizer.decode(output, skip_special_tokens=False)
            
            parsed_triplets = _parse_rebel_output(decoded_text)
            for triplet in parsed_triplets:
                all_triples.append(KnowledgeTriple(
                    subject=triplet['head'],
                    verb=triplet['type'],
                    object=triplet['tail'],
                    extraction_method="rebel",
                    is_deterministic=False # Transformer-based
                ))
            
    return all_triples


class RebelExtractor:
    """Class-based interface for the REBEL extraction engine."""
    def __init__(self):
        # Ensure model is cached
        _load_rebel_model()
        
    def extract(self, text: str) -> list[KnowledgeTriple]:
        """Main extraction method for raw text."""
        return _extract_rebel_triples(text)


def extract_markdown_triples(markdown_text: str) -> list[KnowledgeTriple]:
    """
    Parses Markdown text and extracts triples while preserving hierarchical context
    via structural triples linking headers to subjects.
    """
    # Sanitize incoming PDF text artifacts
    markdown_text = _sanitize_pdf_text(markdown_text)
    
    lines = markdown_text.splitlines()
    active_header = "Document Root"
    grouped_text: list[str] = []
    all_triples: list[KnowledgeTriple] = []

    def process_chunk(header: str, text_lines: list[str]):
        if not text_lines:
            return
        
        chunk_text = "\n".join(text_lines).strip()
        if not chunk_text:
            return
            
        # PIVOT: Use REBEL for extraction instead of spaCy SVO
        rebel_triples = _extract_rebel_triples(chunk_text)
        
        for triple in rebel_triples:
            resolved_subject = triple.subject
            # Simple pronoun resolution: resolve 'This', 'They', 'It' to header context
            if triple.subject.lower() in {"this", "they", "it", "these"} and header != "Document Root":
                prefix = header
                if header.lower() in {"family", "description", "history", "origin"}:
                    prefix = f"Apple {header}"
                
                resolved_subject = f"{prefix} ({triple.subject})"
                # Update triple with resolved subject
                triple = KnowledgeTriple(
                    subject=resolved_subject,
                    verb=triple.verb,
                    object=triple.object,
                    extraction_method=triple.extraction_method,
                    is_deterministic=triple.is_deterministic
                )
                
            all_triples.append(triple)
            
            # Create structural linkage between header and extracted subject
            all_triples.append(KnowledgeTriple(
                subject=header,
                verb="contains concept",
                object=resolved_subject,
                extraction_method="structural",
                is_deterministic=True
            ))

    def clean_header(h):
        h = h.replace("*", "").replace("_", "").strip()
        if h.lower().replace(" ", "") == "family":
            return "Family"
        return h

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        if stripped.startswith("#"):
            process_chunk(clean_header(active_header), grouped_text)
            grouped_text = []
            active_header = stripped.lstrip("#").strip()
        else:
            grouped_text.append(line)

    process_chunk(clean_header(active_header), grouped_text)
    return all_triples


def extract_source_triples(text: str) -> list[KnowledgeTriple]:
    """
    Pivot entry point: Open-world extraction using REBEL with Markdown hierarchy preservation.
    """
    # Swap it into the pipeline
    triples = extract_markdown_triples(text)
    return _deduplicate(triples)


def extract_claim_triples(text: str) -> list[KnowledgeTriple]:
    """
    Closed-world claim extraction. Now uses spaCy SVO as the primary
    extractor since structured claims come directly from the LLM via
    the CLAIMS JSON block in app.py. This function serves as a fallback.
    """
    # Skip heavy parsing for very short texts
    if not text or len(text.split()) < 3:
        return []
    return _deduplicate(_extract_svo_triples(text))


def _deduplicate(triples: list[KnowledgeTriple]) -> list[KnowledgeTriple]:
    """Removes duplicate triples based on subject, verb, and object content."""
    seen: set[tuple[str, str, str]] = set()
    unique: list[KnowledgeTriple] = []
    for t in triples:
        key = (t.subject.strip().lower(),
               t.verb.strip().lower(),
               t.object.strip().lower())
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique

@lru_cache(maxsize=1)
def _load_spacy_model():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

def _extract_svo_triples(text: str) -> list[KnowledgeTriple]:
    """
    Open-world SVO extraction using spaCy.
    """
    nlp = _load_spacy_model()
    doc = nlp(text)
    triples: list[KnowledgeTriple] = []

    for sent in doc.sents:
        subject = _find_subject(sent)
        verb_token = _find_root_verb_token(sent)
        verb = _verb_text(verb_token)
        obj = _find_object(sent, verb_token)

        if subject and verb and obj:
            combined_triple_text = f"{subject} {verb} {obj}".lower()
            orphaned_dates = tuple(
                ent.text.strip() for ent in sent.ents
                if ent.label_ in {"DATE", "TIME"} and ent.text.lower() not in combined_triple_text
            )

            PREPOSITIONS = {"in", "on", "at", "from", "to", "by", "with", "for", "into", "of"}
            obj_parts = obj.split()
            if len(obj_parts) > 1 and obj_parts[0].lower() in PREPOSITIONS:
                prep = obj_parts[0]
                verb = f"{verb} {prep}"
                obj = " ".join(obj_parts[1:])

            triples.append(KnowledgeTriple(
                subject=subject,
                verb=verb,
                object=obj,
                temporal_anchors=orphaned_dates,
                extraction_method="spacy",
                is_deterministic=True
            ))
    return triples

def _find_subject(sent) -> str:
    for token in sent:
        if token.dep_ in {"nsubj", "nsubjpass"}:
            return _span_text(token)
    return ""

def _find_root_verb_token(sent):
    for token in sent:
        if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}:
            return token
    return None

def _verb_text(verb_token) -> str:
    if verb_token is None:
        return ""
    return verb_token.text.strip()

def _find_object(sent, verb_token) -> str:
    if verb_token is None:
        return ""

    seen_ids = set()
    object_tokens = []

    # 1. Direct objects and attributes
    for token in sent:
        if token.dep_ in {"dobj", "obj", "attr", "oprd"} and token.head == verb_token:
            for t in token.subtree:
                if t.i not in seen_ids:
                    seen_ids.add(t.i)
                    object_tokens.append(t)

    # 2. Prepositional phrases attached to the verb
    for child in verb_token.children:
        if child.dep_ == "prep":
            for t in child.subtree:
                if t.i not in seen_ids:
                    seen_ids.add(t.i)
                    object_tokens.append(t)

    if not object_tokens:
        return ""

    # Sort by original token index to maintain sentence order
    object_tokens.sort(key=lambda x: x.i)
    return " ".join(t.text for t in object_tokens).strip()

def _span_text(token) -> str:
    subtree = getattr(token, "subtree", None)
    if subtree is not None:
        try:
            text = " ".join(node.text for node in subtree).strip()
            if text:
                return text
        except TypeError:
            text = getattr(subtree, "text", "").strip()
            if text:
                return text

    return getattr(token, "text", "").strip()
