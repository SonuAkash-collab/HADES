from __future__ import annotations

import re
import spacy
from functools import lru_cache
from dataclasses import dataclass

from cerberus.core.source_graph import SourceGraph
from shared.triple import KnowledgeTriple


@dataclass(slots=True)
class VerificationResult:
    is_verified: bool
    reason: str
    label: str = ""




def verify_claim(claim: KnowledgeTriple, source_graph: SourceGraph, model_name: str = "cross-encoder/nli-deberta-v3-base", source_sentences: list[str] = None) -> VerificationResult:
    import torch
    import torch.nn.functional as F

    premise = _build_localized_premise(claim, source_graph, source_sentences)
    if not premise:
        return VerificationResult(is_verified=False, reason="No relevant facts found in the source graph context")

    tokenizer, model = _load_nli_model(model_name)
    inputs = tokenizer(
        premise,
        claim.as_text(),
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = int(torch.argmax(outputs.logits, dim=-1).item())

    label = _resolve_label(model, prediction)
    
    # print(f"CLAIM: {claim.as_text()}")
    # print(f"PREMISE: {premise}")
    # print(f"PREDICTION: {label}")

    if label == "neutral" and source_sentences:
        # Retry with progressively more context (top-2, then top-3 sentences)
        claim_keywords = _get_claim_keywords(claim.as_text())
        scored = []
        for sentence in source_sentences:
            overlap = len(claim_keywords.intersection(_get_claim_keywords(sentence)))
            if overlap >= 1:
                scored.append((overlap, " ".join(sentence.split())))
        scored.sort(key=lambda x: x[0], reverse=True)

        for n in [2, 3]:
            if len(scored) < n:
                break
            extended_premise = " ".join(s for _, s in scored[:n])
            ext_inputs = tokenizer(
                extended_premise,
                claim.as_text(),
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            with torch.no_grad():
                ext_outputs = model(**ext_inputs)
                ext_prediction = int(torch.argmax(ext_outputs.logits, dim=-1).item())
            ext_label = _resolve_label(model, ext_prediction)
            if ext_label == "entailment":
                probs = F.softmax(ext_outputs.logits, dim=-1)
                entail_idx = [
                    i for i, l in model.config.id2label.items()
                    if "entail" in l.lower()
                ][0]
                conf = probs[0][entail_idx].item()
                if conf > 0.70:
                    label = "entailment"
                    prediction = ext_prediction
                    outputs = ext_outputs
                    break

    if not claim.is_deterministic and label == "entailment":
        probs = F.softmax(outputs.logits, dim=-1)
        entailment_idx = [i for i, l in model.config.id2label.items() if "entail" in l.lower()][0]
        entailment_score = probs[0][entailment_idx].item()

        if entailment_score <= 0.85:
            return VerificationResult(
                is_verified=False,
                reason=f"GLiNER-extracted triple requires higher confidence threshold (got {entailment_score:.2f})",
                label="neutral"
            )

    # GPE precision check — prevent geographic over-generalisation
    # e.g. "United Kingdom" claimed when source says "England"
    if label == "entailment":
        nlp = _load_spacy()
        claim_doc = nlp(claim.as_text())
        claim_gpes = {
            ent.text.lower().strip()
            for ent in claim_doc.ents
            if ent.label_ in ("GPE", "LOC")
        }
        if claim_gpes and source_sentences:
            # Check if all claimed GPEs appear in at least one source sentence
            source_text = " ".join(source_sentences).lower()
            missing_gpes = {
                gpe for gpe in claim_gpes
                if gpe not in source_text
            }
            if missing_gpes:
                return VerificationResult(
                    is_verified=False,
                    reason=(
                        f"Geographic precision check failed: "
                        f"{missing_gpes} not found in source document."
                    ),
                    label="neutral"
                )

    if label == "entailment":
        return VerificationResult(
            is_verified=True,
            reason="Verified by local DeBERTa-v3 NLI model against the source graph.",
            label=label,
        )

    if label == "contradiction":
        reason = "Rejected by local DeBERTa-v3 NLI model: the claim contradicts the source graph."
    else:
        reason = f"Rejected by local DeBERTa-v3 NLI model: the claim is not entailed by the source graph (label: {label})."

    return VerificationResult(is_verified=False, reason=reason, label=label)



@lru_cache(maxsize=1)
def _load_spacy():
    return spacy.load("en_core_web_sm")


def _get_claim_keywords(text: str) -> set[str]:
    """
    Extract keywords from a claim for premise retrieval.
    Only uses lemmas to avoid duplicate counts for plurals/forms.
    """
    nlp = _load_spacy()
    doc = nlp(text.lower())
    keywords = set()
    
    for token in doc:
        # Keep: nouns, proper nouns, numbers
        if token.pos_ in {"NOUN", "PROPN", "NUM"}:
            keywords.add(token.lemma_)
        # Keep: important adjectives (e.g. "poisonous")
        elif token.pos_ == "ADJ" and len(token.text) > 4:
            keywords.add(token.lemma_)
        # Keep: verbs (lemmatised)
        elif token.pos_ == "VERB" and len(token.text) > 1:
            keywords.add(token.lemma_)
        # Keep: any other long meaningful words
        elif len(token.text) > 3 and not token.is_punct and not token.is_stop:
            keywords.add(token.lemma_)
            
    return keywords


def _build_localized_premise(claim: KnowledgeTriple, source_graph: SourceGraph, source_sentences: list[str] = None) -> str:
    claim_keywords = _get_claim_keywords(claim.as_text())
    
    # Calculate PageRank scores for tie-breaking
    import networkx as nx
    try:
        # Use PageRank from nodes
        scores = nx.pagerank(source_graph.graph, weight="weight")
    except Exception:
        scores = {}

    # Match triples and score them
    scored_triples: list[tuple[KnowledgeTriple, int, float]] = []
    for triple in source_graph.triples:
        triple_keywords = _get_claim_keywords(triple.as_text())
        overlap = len(claim_keywords.intersection(triple_keywords))
        if overlap >= 1:
            # Score = subject_pr + object_pr
            pr_score = scores.get(triple.subject, 0.0) + scores.get(triple.object, 0.0)
            scored_triples.append((triple, overlap, pr_score))
    
    # Sort by overlap DESC, then PageRank DESC
    scored_triples.sort(key=lambda x: (x[1], x[2]), reverse=True)
    matching_triples = [t for t, _, _ in scored_triples]

    # Match sentences  
    scored_sentences: list[tuple[int, str]] = []
    if source_sentences:
        for sentence in source_sentences:
            sent_keywords = _get_claim_keywords(sentence)
            overlap = len(claim_keywords.intersection(sent_keywords))
            if overlap >= 1:
                clean_sent = " ".join(sentence.split())
                scored_sentences.append((overlap, clean_sent))
    
    import re as _re_num
    claim_numbers = set(_re_num.findall(
        r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b|\b\d+(?:\.\d+)?%?\b', claim.as_text()
    ))
    if claim_numbers:
        boosted_sentences = []
        for score, sentence in scored_sentences:
            sentence_numbers = set(_re_num.findall(
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b|\b\d+(?:\.\d+)?%?\b', sentence
            ))
            if sentence_numbers:
                score += 3
            boosted_sentences.append((score, sentence))
        scored_sentences = boosted_sentences

    # Sort by overlap score descending, take only the TOP 1
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    matching_sentences = [s for _, s in scored_sentences[:1]]
    
    # Build premise: prose first, then triples
    # Limit to 1 sentences and 10 triples to ensure full coverage (increased from 8)
    premise_parts = []
    for sent in matching_sentences:
        premise_parts.append(sent)
    for triple in matching_triples[:10]:
        clean_triple = " ".join(triple.as_text().split())
        premise_parts.append(clean_triple)
    
    premise = " ".join(premise_parts)
    return " ".join(premise.split())  # final whitespace normalisation




@lru_cache(maxsize=1)
def _load_nli_model(model_name: str):
    import streamlit as st
    
    # NEW LOGGING STATEMENTS
    # Removed for final cleanup
            
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _resolve_label(model, prediction: int) -> str:
    id2label = getattr(model.config, "id2label", {}) or {}
    label = str(id2label.get(prediction, "")).lower()
    if "entail" in label:
        return "entailment"
    if "contrad" in label:
        return "contradiction"
    if "neutral" in label:
        return "neutral"
    if prediction == 2:
        return "entailment"
    if prediction == 0:
        return "contradiction"
    return "neutral"
