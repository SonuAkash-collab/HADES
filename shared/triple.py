from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KnowledgeTriple:
    subject: str
    verb: str
    object: str
    temporal_anchors: tuple[str, ...] = tuple()
    modality: str = ""
    is_negated: bool = False
    condition: str = ""
    extraction_method: str = "spacy"
    is_deterministic: bool = True

    def as_text(self) -> str:
        """
        Formats the triple as clean subject-verb-object text.
        Avoids grammatical inversions that confuse small LLMs.
        """
        parts = [
            p.strip() for p in [self.subject, self.verb, self.object]
            if p and p.strip()
        ]
        text = " ".join(parts)

        if self.is_negated:
            text = f"NOT: {text}"
        if self.temporal_anchors:
            text += f" [{', '.join(self.temporal_anchors)}]"
        if self.condition:
            text += f" (if {self.condition})"

        return text
