# HADES Copilot Instructions

You are helping maintain HADES (Hierarchical Adaptive Document Encoding System) as a two-path research repo.

## Project Structure
- `shared/` is the single source of truth for reusable logic.
- `charon/` (Charon) is Path A: compression, PageRank scoring, and L1 context handling.
- `cerberus/` (Cerberus) is Path B: source-graph verification and L3 persistence.

## Non-Negotiable Rules
- Local-only extraction: use spaCy with `en_core_web_sm` for all SVO triple extraction.
- Local-only verification: use `cross-encoder/nli-deberta-v3-small` for all NLI checks.
- No GPT for extraction.
- No GPT for verification.
- Cerberus verifies claims against a pre-ingested source graph, not against previous AI responses.
- Shared logic must come from `shared/`, not duplicated in Charon (charon/) or Cerberus (cerberus/).
- Treat the legacy root-level prototype files as deprecated; prefer the package layout under `shared/`, `charon/`, and `cerberus/`.

## Ownership Map
- `shared/triple.py`: `KnowledgeTriple` dataclass.
- `shared/extractor.py`: spaCy-based SVO extraction.
- `charon/core/graph.py`: NetworkX graph construction and PageRank scoring.
- `charon/core/cache.py`: token-budgeted L1 cache.
- `charon/core/compressor.py`: compression and prose generation helpers.
- `cerberus/core/source_graph.py`: builds the source-of-truth graph from ingested documents.
- `cerberus/core/verifier.py`: deterministic NLI verification.
- `cerberus/core/wiki_storage.py`: JSON persistence for verified facts.

## Generation Rules
- When adding extraction logic, import and extend `shared/extractor.py` instead of recreating parsing code elsewhere.
- When adding verification logic, route through `cerberus/core/verifier.py` and keep the model local.
- When adding common data structures, place them in `shared/` first and import them from both sub-projects.
- When adding persistence, put it under `cerberus/core/` so Cerberus owns the L3 store.

## Behavior Rules
- Prefer deterministic, local computation over external API calls.
- Keep imports aligned with the new folder structure.
- Do not regenerate legacy root-level code paths unless a migration task explicitly asks for backward compatibility.
- If a task conflicts with these rules, follow the rules above and call out the conflict in your response.