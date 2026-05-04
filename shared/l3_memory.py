from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.triple import KnowledgeTriple


_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = _WORKSPACE_ROOT / "cerberus" / "core" / "wiki.db"
DEFAULT_LEGACY_JSON_PATH = _WORKSPACE_ROOT / "cerberus" / "core" / "wiki.json"
_DB_LOCK = threading.RLock()
_INITIALIZED_DATABASES: set[Path] = set()

_CREATE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_base (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    verb TEXT NOT NULL,
    object TEXT NOT NULL,
    modality TEXT NOT NULL DEFAULT '',
    is_negated INTEGER NOT NULL DEFAULT 0,
    condition TEXT NOT NULL DEFAULT '',
    temporal_anchors TEXT NOT NULL DEFAULT '[]',
    source_page INTEGER,
    cerberus_status TEXT NOT NULL CHECK (cerberus_status IN ('CLEAN', 'DIRTY')),
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(subject, verb, object, condition, temporal_anchors)
);
"""

_INSERT_FACT_SQL = """
INSERT OR IGNORE INTO knowledge_base
    (subject, verb, object, modality, is_negated, condition, temporal_anchors, source_page, cerberus_status, timestamp)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


def initialize_l3_memory(
    db_path: str | Path | None = None,
    legacy_json_path: str | Path | None = None,
) -> Path:
    resolved_db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    resolved_legacy_json_path = (
        Path(legacy_json_path) if legacy_json_path is not None else DEFAULT_LEGACY_JSON_PATH
    )
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)

    with _DB_LOCK:
        if resolved_db_path not in _INITIALIZED_DATABASES:
            with _connect(resolved_db_path) as conn:
                conn.execute(_CREATE_SCHEMA_SQL)
                conn.commit()

            _migrate_legacy_json(
                db_path=resolved_db_path,
                legacy_json_path=resolved_legacy_json_path,
            )
            _INITIALIZED_DATABASES.add(resolved_db_path)

    return resolved_db_path


def save_fact(
    subject: str | KnowledgeTriple,
    verb: str | None = None,
    object_text: str | None = None,
    *,
    source_page: int | None = None,
    cerberus_status: str = "CLEAN",
    db_path: str | Path | None = None,
) -> bool:
    triple = (
        subject
        if isinstance(subject, KnowledgeTriple)
        else KnowledgeTriple(subject, verb or "", object_text or "")
    )

    normalized_subject = str(triple.subject).strip()
    normalized_verb = str(triple.verb).strip()
    normalized_object = str(triple.object).strip()
    if not (normalized_subject and normalized_verb and normalized_object):
        return False

    normalized_status = str(cerberus_status).strip().upper() or "CLEAN"
    if normalized_status not in {"CLEAN", "DIRTY"}:
        raise ValueError("cerberus_status must be CLEAN or DIRTY")

    normalized_source_page = _normalize_source_page(source_page)
    resolved_db_path = initialize_l3_memory(db_path=db_path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with _DB_LOCK:
        with _connect(resolved_db_path) as conn:
            # Check for existing facts with same subject+verb but different object
            existing = conn.execute(
                """SELECT object FROM knowledge_base
                   WHERE subject = ? AND verb = ? AND cerberus_status = 'CLEAN'""",
                (normalized_subject, normalized_verb)
            ).fetchall()

            if existing:
                existing_objects = [row["object"] for row in existing]
                if normalized_object in existing_objects:
                    return False  # exact duplicate, skip silently
                # Conflicting fact — different object for same subject+verb
                # Log the conflict but do not write; caller can decide
                import logging
                logging.warning(
                    f"[L3 CONFLICT] Skipping write: '{normalized_subject} "
                    f"{normalized_verb} {normalized_object}' conflicts with "
                    f"existing: {existing_objects}"
                )
                return False

            # No conflict — safe to write
            cursor = conn.execute(
                _INSERT_FACT_SQL,
                (
                    normalized_subject,
                    normalized_verb,
                    normalized_object,
                    triple.modality,
                    1 if triple.is_negated else 0,
                    triple.condition,
                    json.dumps(triple.temporal_anchors),
                    normalized_source_page,
                    normalized_status,
                    now,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0


def fetch_clean_facts(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    resolved_db_path = initialize_l3_memory(db_path=db_path)

    with _DB_LOCK:
        with _connect(resolved_db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, subject, verb, object, modality, is_negated, condition, 
                       temporal_anchors, source_page, cerberus_status, timestamp
                FROM knowledge_base
                WHERE cerberus_status = 'CLEAN'
                ORDER BY id ASC
                """
            ).fetchall()

    results = []
    for row in rows:
        d = dict(row)
        try:
            d["temporal_anchors"] = tuple(json.loads(d.get("temporal_anchors", "[]")))
        except (json.JSONDecodeError, TypeError):
            d["temporal_anchors"] = tuple()
        results.append(d)

    return results


def _migrate_legacy_json(db_path: Path, legacy_json_path: Path) -> int:
    if not legacy_json_path.exists():
        return 0

    rows_to_insert: list[tuple[str, str, str, int, str, str]] = []
    raw_text = legacy_json_path.read_text(encoding="utf-8").strip()
    if raw_text:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = []

        if isinstance(payload, list):
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for item in payload:
                if not isinstance(item, dict):
                    continue

                subject = str(item.get("subject", "")).strip()
                verb = str(item.get("verb", "")).strip()
                object_text = str(item.get("object", "")).strip()
                if not (subject and verb and object_text):
                    continue

                status = str(item.get("cerberus_status", "CLEAN")).strip().upper() or "CLEAN"
                if status not in {"CLEAN", "DIRTY"}:
                    status = "CLEAN"

                source_page = _normalize_source_page(item.get("source_page"))
                timestamp = str(item.get("timestamp", "")).strip() or now
                
                modality = str(item.get("modality", "")).strip()
                is_negated = 1 if item.get("is_negated") else 0
                condition = str(item.get("condition", "")).strip()
                temporal_anchors = json.dumps(item.get("temporal_anchors", []))
                
                rows_to_insert.append((
                    subject, verb, object_text, modality, is_negated, condition, 
                    temporal_anchors, source_page, status, timestamp
                ))

    if rows_to_insert:
        with _connect(db_path) as conn:
            conn.executemany(_INSERT_FACT_SQL, rows_to_insert)
            conn.commit()

    backup_path = legacy_json_path.with_name("wiki_old.json.backup")
    if backup_path.exists():
        backup_path.unlink()
    legacy_json_path.replace(backup_path)
    return len(rows_to_insert)


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path), timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    return connection


def _normalize_source_page(raw_value: Any) -> int:
    if raw_value is None:
        return 0

    try:
        page_number = int(raw_value)
    except (TypeError, ValueError):
        return 0

    return page_number if page_number >= 0 else 0


def fetch_clean_facts_by_similarity(
    keyword: str,
    embedder,
    threshold: float = 0.60,
    limit: int = 5,
) -> list[dict]:
    from sklearn.metrics.pairwise import cosine_similarity

    # 1. Load all CLEAN facts
    facts = fetch_clean_facts()
    if not facts:
        return []

    # 2. Build fact strings for similarity
    fact_strings = []
    for f in facts:
        text = f"{f['subject']} {f['verb']} {f['object']}"
        if f.get("is_negated"):
            text = f"{f['subject']} not {f['verb']} {f['object']}"
        if f.get("modality"):
            text = f"{text} ({f['modality']})"
        if f.get("condition"):
            text = f"{text} (Condition: {f['condition']})"
        fact_strings.append(text)

    # 3. Batch encode fact strings
    fact_vectors = embedder.encode(fact_strings)

    # 4. Encode keyword
    keyword_vector = embedder.encode(keyword)

    # 5. Compute cosine similarity
    scores = cosine_similarity([keyword_vector], fact_vectors)[0]

    # 6. Filter and attach scores
    scored_facts = []
    for i, score in enumerate(scores):
        val = float(score)
        if val >= threshold:
            fact = dict(facts[i])
            fact["similarity_score"] = val
            scored_facts.append(fact)

    # 7. Sort by score descending and limit
    scored_facts.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored_facts[:limit]