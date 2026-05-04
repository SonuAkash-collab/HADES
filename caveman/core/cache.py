from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable
import tiktoken

from shared.triple import KnowledgeTriple


_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass(slots=True)
class CacheEntry:
    triple: KnowledgeTriple
    pagerank_score: float = 0.0

    @property
    def text(self) -> str:
        return self.triple.as_text()


@dataclass(slots=True)
class HistoryTurn:
    role: str
    text: str


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    text: str


DEFAULT_BUDGETS: dict[str, int] = {
    "system": 50,
    "facts": 100,
    "history": 150,
    "tools": 100,
    "scratch": 80,
}

MIN_FACTS_FLOOR: int = 2


# Routing rules that map triples to their correct set.
# Each rule is a predicate on the triple's verb and subject.
# First matching rule wins. "facts" is the default fallback.
SET_ROUTING_RULES: dict[str, object] = {
    "history": lambda t: any(
        w in t.verb.lower()
        for w in ["said", "spoke", "asked", "replied", "told",
                  "reported", "announced", "stated", "claimed"]
    ),
    "tools": lambda t: any(
        w in t.subject.lower()
        for w in ["tool", "result", "output", "search",
                  "query", "api", "response"]
    ),
}


class L1Cache:
    def __init__(self, budgets: dict[str, int] | None = None) -> None:
        self.budgets = self._resolved_budgets(budgets)

        self.set_system: list[str] = []
        self.set_facts: OrderedDict[str, CacheEntry] = OrderedDict()
        self.set_history: deque[HistoryTurn] = deque()
        self.set_tools: deque[ToolResult] = deque()
        self.set_scratch: deque[str] = deque()

    def add_system_instruction(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned or cleaned in self.set_system:
            return

        projected = self._estimate_system_tokens() + _count_tokens(cleaned)
        if projected > self.budgets["system"]:
            raise ValueError("System set budget exceeded; system instructions are pinned and cannot be evicted.")

        self.set_system.append(cleaned)

    def add_fact(
        self,
        triple: KnowledgeTriple,
        pagerank_score: float = 0.0,
    ) -> None:
        if not isinstance(triple, KnowledgeTriple):
            raise TypeError("add_fact expects a KnowledgeTriple instance.")

        entry = CacheEntry(triple=triple, pagerank_score=pagerank_score)
        self.set_facts[entry.text] = entry
        self._trim_facts_to_budget()

    def add_history_turn(self, role: str, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return

        self.set_history.append(HistoryTurn(role=role.strip() or "user", text=cleaned))
        self._trim_history_to_budget()

    def add_tool_result(self, tool_name: str, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return

        self.set_tools.append(ToolResult(tool_name=tool_name.strip() or "tool", text=cleaned))
        self._trim_tools_to_budget()

    def add_scratch_entry(self, text: str) -> None:
        """
        Add LLM-generated content to the SCRATCH set.
        All content here is dirty (unverified) by default.
        It will be evicted and passed to Sentinel before 
        write-back to L2/L3.
        """
        cleaned = text.strip()
        if not cleaned:
            return
        self.set_scratch.append(cleaned)
        self._trim_scratch_to_budget()

    def route_triple(
        self,
        triple: KnowledgeTriple,
        pagerank_score: float = 0.0,
    ) -> str:
        """
        Automatically route a triple to the correct set based on
        its semantic type. Implements the set-index addressing
        function from set-associative cache architecture.

        Returns the name of the set it was routed to.

        The routing function works as follows:
        - Speech/communication verbs → HISTORY set
        - Tool/API/output subjects → TOOLS set
        - Everything else → FACTS set (default)

        This ensures that different semantic categories of knowledge
        occupy separate cache partitions, preventing eviction
        competition between structurally unrelated facts.
        """
        for set_name, rule in SET_ROUTING_RULES.items():
            if rule(triple):
                if set_name == "history":
                    self.add_history_turn("system", triple.as_text())
                elif set_name == "tools":
                    self.add_tool_result("auto", triple.as_text())
                return set_name

        # Default route: FACTS
        self.add_fact(triple, pagerank_score=pagerank_score)
        return "facts"

    def get_routing_stats(self) -> dict[str, int]:
        """Return count of entries in each set for telemetry."""
        return {
            "system": len(self.set_system),
            "facts": len(self.set_facts),
            "history": len(self.set_history),
            "tools": len(self.set_tools),
            "scratch": len(self.set_scratch),
        }

    def rerank_facts_for_query(
        self,
        query: str,
        embedder,
        alpha: float = 0.4,
    ) -> None:
        """
        Re-rank FACTS set by blending PageRank score with cosine
        similarity to the user query.

        This implements query-aware cache prefetching: once the
        query is known, facts most relevant to it are promoted
        (given higher scores) so they survive budget trimming.

        Formula:
            new_score = alpha * pagerank_score
                      + (1 - alpha) * cosine_similarity(fact, query)

        alpha=0.4 weights query relevance (0.6) more heavily than
        structural importance (0.4). Adjust alpha toward 1.0 to
        rely more on PageRank, toward 0.0 to rely more on query.

        Args:
            query: The user's question string.
            embedder: sentence-transformers SentenceTransformer instance.
            alpha: blend weight for PageRank score (0.0-1.0).
        """
        if not self.set_facts:
            return

        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        import numpy as np

        query_vec = embedder.encode(query).reshape(1, -1)
        
        # Aggressive L1 Filtering: Identify temporal/numeric context of the query
        import re
        query_years = set(re.findall(r'\b(?:19|20)\d{2}\b', query))
        query_numbers = set(re.findall(r'\b\d+(?:[\.,]\d+)?%?\b', query))
        has_query_stats = bool(query_years or query_numbers)

        to_delete = []
        for key, entry in self.set_facts.items():
            fact_text = entry.text
            triple = entry.triple
            
            # Hard Temporal Exclusion: If query has a year, 
            # purge facts with DIFFERENT years in metadata to starve SLM hallucinations
            fact_years = set()
            for anchor in triple.temporal_anchors:
                fact_years.update(re.findall(r'\b(?:19|20)\d{2}\b', anchor))
            
            # Also check text just in case extraction missed something or for legacy compatibility
            fact_years.update(re.findall(r'\b(?:19|20)\d{2}\b', fact_text))
            
            if query_years and fact_years and not (fact_years & query_years):
                to_delete.append(key)
                continue

            fact_vec = embedder.encode(fact_text).reshape(1, -1)
            similarity = float(cos_sim(query_vec, fact_vec)[0][0])
            
            # Numeric/Temporal Filtering Logic
            stat_penalty = 1.0
            fact_numbers = set(re.findall(r'\b\d+(?:[\.,]\d+)?%?\b', fact_text))
            
            if has_query_stats:
                # Penalize facts with different numbers (if not already purged by year)
                mismatched_numbers = fact_numbers - query_numbers
                if mismatched_numbers:
                    stat_penalty = 0.4
            else:
                # If query is general (no numbers), penalize ANY facts with numbers
                if fact_years or fact_numbers:
                    stat_penalty = 0.6

            # Blend structural importance, query relevance, and stat filter
            entry.pagerank_score = (
                (alpha * entry.pagerank_score + (1.0 - alpha) * similarity) 
                * stat_penalty
            )

        # Immediate Purge
        for key in to_delete:
            del self.set_facts[key]

        # Re-trim with updated scores so highest-relevance facts survive
        self._trim_facts_to_budget()

    def _trim_scratch_to_budget(self) -> None:
        while self._estimate_scratch_tokens() > self.budgets["scratch"] \
              and self.set_scratch:
            self.set_scratch.popleft()

    def _estimate_scratch_tokens(self) -> int:
        return sum(_count_tokens(t) for t in self.set_scratch)

    def flush_scratch(self) -> list[str]:
        """
        Evict all SCRATCH entries for Sentinel verification.
        Called by the write-back gate before session ends.
        Returns the list of dirty entries for verification.
        """
        dirty = list(self.set_scratch)
        self.set_scratch.clear()
        return dirty

    def extend(self, triples: Iterable[KnowledgeTriple]) -> None:
        for triple in triples:
            self.add_fact(triple)

    @property
    def active_facts(self) -> OrderedDict[str, CacheEntry]:
        return self.set_facts

    def as_context_lines(self) -> list[str]:
        lines: list[str] = []

        if self.set_system:
            lines.append("System Instructions:")
            lines.extend(self.set_system)

        if self.set_tools:
            lines.append("Active Tool Results:")
            lines.extend(f"{item.tool_name}: {item.text}" for item in self.set_tools)

        if self.set_facts:
            lines.append("Fact Cache:")
            lines.extend(entry.text for entry in self.set_facts.values())

        if self.set_history:
            lines.append("Conversation History:")
            lines.extend(f"{turn.role}: {turn.text}" for turn in self.set_history)

        if self.set_scratch:
            lines.append("Unverified Scratch (Dirty):")
            lines.extend(self.set_scratch)

        return lines

    def as_context_text(self) -> str:
        return "\n".join(self.as_context_lines())

    def _trim_facts_to_budget(self) -> None:
        while (self._estimate_facts_tokens() > self.budgets["facts"]
               and len(self.set_facts) > MIN_FACTS_FLOOR):
            lowest_key = min(
                self.set_facts,
                key=lambda key: self.set_facts[key].pagerank_score
            )
            del self.set_facts[lowest_key]

    def _trim_history_to_budget(self) -> None:
        while self._estimate_history_tokens() > self.budgets["history"] and self.set_history:
            self.set_history.popleft()

    def _trim_tools_to_budget(self) -> None:
        while self._estimate_tools_tokens() > self.budgets["tools"] and self.set_tools:
            self.set_tools.popleft()

    def _estimate_system_tokens(self) -> int:
        return sum(_count_tokens(text) for text in self.set_system)

    def _estimate_facts_tokens(self) -> int:
        return sum(_count_tokens(entry.text) for entry in self.set_facts.values())

    def _estimate_history_tokens(self) -> int:
        return sum(_count_tokens(f"{turn.role}: {turn.text}") for turn in self.set_history)

    def _estimate_tools_tokens(self) -> int:
        return sum(_count_tokens(f"{item.tool_name}: {item.text}") for item in self.set_tools)

    def _resolved_budgets(self, budgets: dict[str, int] | None) -> dict[str, int]:
        resolved = dict(DEFAULT_BUDGETS)
        if budgets is not None:
            unknown = set(budgets).difference(DEFAULT_BUDGETS)
            if unknown:
                unknown_names = ", ".join(sorted(unknown))
                raise ValueError(f"Unknown budget set(s): {unknown_names}")
            resolved.update(budgets)

        for set_name, value in resolved.items():
            if value < 0:
                raise ValueError(f"Budget for '{set_name}' must be non-negative.")

        return resolved


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))
