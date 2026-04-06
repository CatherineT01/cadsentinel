"""
cadsentinel.etl.retriever
--------------------------
Evidence retrieval engine.

Given a spec rule and a drawing_id, executes the rule's retrieval_recipe
against the database and returns a bounded evidence package ready for the
spec execution engine.

Two-stage retrieval:
    Stage 1 — Metadata filter (SQL, fast, exact)
        Queries drawing_layers, drawing_dimensions, drawing_title_block,
        and drawing_entities by layer pattern, category, and entity type.

    Stage 2 — Vector search (pgvector cosine similarity, semantic)
        Embeds the spec rule text and finds the most similar text chunks
        in drawing_text_chunks for this drawing.

Results are merged, deduplicated by entity handle, and trimmed to top_k.

Usage:
    retriever = EvidenceRetriever()
    package = retriever.retrieve(
        drawing_id=7,
        spec_rule_id=142,
        normalized_rule_text="All dimensions must be in inches.",
        retrieval_recipe={
            "source_types": ["note", "dimension", "title_block"],
            "top_k": 8,
            "keyword_filters": ["inch", "units"],
            "entity_filters": {"layers": ["DIM*", "ANNO*"]}
        },
        checker_prompt=None,   # optional — used as vector query if present
    )
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .db import get_connection

log = logging.getLogger(__name__)

EMBEDDING_MODEL = os.environ.get("CADSENTINEL_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
CHROMA_PATH = os.environ.get("CADSENTINEL_CHROMA_PATH", "./chroma_db")

# Source type constants — match retrieval_recipe source_types values
SOURCE_NOTE       = "note"
SOURCE_DIMENSION  = "dimension"
SOURCE_TITLE      = "title_block"
SOURCE_LAYER      = "layer"
SOURCE_ENTITY     = "entity"


class RetrievalError(RuntimeError):
    """Raised when retrieval cannot proceed."""


class EvidenceRetriever:
    """
    Executes retrieval_recipe queries against the database and returns
    a structured evidence package for one spec rule + one drawing.
    """

    def __init__(self, embed_for_vector_search: bool = True):
        """
        Args:
            embed_for_vector_search: If False, skip Stage 2 vector search.
                Useful for deterministic-only specs or when no embeddings exist yet.
        """
        self.embed_for_vector_search = embed_for_vector_search

    # ── Public API ──────────────────────────────────────────────── #

    def retrieve(
        self,
        drawing_id: int,
        spec_rule_id: int,
        normalized_rule_text: str,
        retrieval_recipe: dict,
        checker_prompt: Optional[str] = None,
    ) -> dict:
        """
        Execute retrieval for one spec rule against one drawing.

        Returns an evidence package dict ready for the execution engine.
        """
        source_types   = retrieval_recipe.get("source_types", [])
        top_k          = int(retrieval_recipe.get("top_k", 8))
        keyword_filters = retrieval_recipe.get("keyword_filters", [])
        entity_filters = retrieval_recipe.get("entity_filters", {})
        layer_patterns = entity_filters.get("layers", [])

        evidence: list[dict] = []

        with get_connection() as conn:
            with conn.cursor() as cur:

                # ── Stage 1: metadata filter ─────────────────────── #

                if SOURCE_NOTE in source_types or SOURCE_ENTITY in source_types:
                    rows = self._fetch_text_entities(
                        cur, drawing_id, layer_patterns, keyword_filters
                    )
                    evidence.extend(rows)

                if SOURCE_DIMENSION in source_types:
                    rows = self._fetch_dimensions(
                        cur, drawing_id, layer_patterns
                    )
                    evidence.extend(rows)

                if SOURCE_TITLE in source_types:
                    row = self._fetch_title_block(cur, drawing_id)
                    if row:
                        evidence.append(row)

                if SOURCE_LAYER in source_types:
                    rows = self._fetch_layers(
                        cur, drawing_id, layer_patterns
                    )
                    evidence.extend(rows)

                # ── Stage 2: vector search ───────────────────────── #

                if self.embed_for_vector_search:
                    query_text = checker_prompt or normalized_rule_text
                    vector_rows = self._vector_search(
                        cur, drawing_id, query_text, top_k
                    )
                    evidence.extend(vector_rows)

        # ── Merge, deduplicate, trim ─────────────────────────────── #

        evidence = self._deduplicate(evidence)
        evidence = self._rank(evidence)
        truncated = len(evidence) > top_k
        evidence = evidence[:top_k]

        return {
            "spec_rule_id":     spec_rule_id,
            "drawing_id":       drawing_id,
            "retrieval_method": "hybrid" if self.embed_for_vector_search else "metadata",
            "query_text":       checker_prompt or normalized_rule_text,
            "evidence":         evidence,
            "evidence_count":   len(evidence),
            "truncated":        truncated,
        }

    # ── Stage 1 helpers ─────────────────────────────────────────── #

    def _fetch_text_entities(
        self,
        cur,
        drawing_id: int,
        layer_patterns: list[str],
        keyword_filters: list[str],
    ) -> list[dict]:
        """Fetch TEXT/MTEXT entities, optionally filtered by layer and keywords."""

        params: list[Any] = [drawing_id]
        sql = """
            SELECT
                entity_type,
                handle,
                layer,
                text_content,
                geometry_json
            FROM drawing_entities
            WHERE drawing_id = %s
              AND category = 'text'
              AND text_content IS NOT NULL
        """

        if layer_patterns:
            pattern_clauses = " OR ".join(
                "layer ILIKE %s" for _ in layer_patterns
            )
            sql += f" AND ({pattern_clauses})"
            for p in layer_patterns:
                params.append(p.replace("*", "%"))

        if keyword_filters:
            kw_clauses = " OR ".join(
                "text_content ILIKE %s" for _ in keyword_filters
            )
            sql += f" AND ({kw_clauses})"
            for kw in keyword_filters:
                params.append(f"%{kw}%")

        sql += " LIMIT 50"

        cur.execute(sql, params)
        rows = cur.fetchall()

        return [
            {
                "source":        "drawing_entities",
                "entity_type":   r["entity_type"],
                "entity_handle": r["handle"],
                "layer":         r["layer"],
                "text":          r["text_content"],
                "geometry":      r["geometry_json"],
                "similarity_score": None,
            }
            for r in rows
        ]

    def _fetch_dimensions(
        self,
        cur,
        drawing_id: int,
        layer_patterns: list[str],
    ) -> list[dict]:
        """Fetch dimension records, optionally filtered by layer."""

        params: list[Any] = [drawing_id]
        sql = """
            SELECT
                dim_type,
                handle,
                layer,
                measured_value,
                user_text,
                text_position_json,
                geometry_json
            FROM drawing_dimensions
            WHERE drawing_id = %s
        """

        if layer_patterns:
            pattern_clauses = " OR ".join(
                "layer ILIKE %s" for _ in layer_patterns
            )
            sql += f" AND ({pattern_clauses})"
            for p in layer_patterns:
                params.append(p.replace("*", "%"))

        sql += " LIMIT 50"

        cur.execute(sql, params)
        rows = cur.fetchall()

        return [
            {
                "source":          "drawing_dimensions",
                "dim_type":        r["dim_type"],
                "entity_handle":   r["handle"],
                "layer":           r["layer"],
                "measured_value":  r["measured_value"],
                "user_text":       r["user_text"],
                "text_position":   r["text_position_json"],
                "geometry":        r["geometry_json"],
                "similarity_score": None,
            }
            for r in rows
        ]

    def _fetch_title_block(self, cur, drawing_id: int) -> Optional[dict]:
        """Fetch the title block record for this drawing."""
        cur.execute(
            """
            SELECT
                found,
                block_name,
                handle,
                layer,
                attributes_json,
                geometry_json,
                detection_confidence
            FROM drawing_title_block
            WHERE drawing_id = %s
            """,
            (drawing_id,),
        )
        r = cur.fetchone()
        if not r or not r["found"]:
            return None

        return {
            "source":               "drawing_title_block",
            "block_name":           r["block_name"],
            "entity_handle":        r["handle"],
            "layer":                r["layer"],
            "attributes":           r["attributes_json"],
            "geometry":             r["geometry_json"],
            "detection_confidence": r["detection_confidence"],
            "similarity_score":     None,
        }

    def _fetch_layers(
        self,
        cur,
        drawing_id: int,
        layer_patterns: list[str],
    ) -> list[dict]:
        """Fetch layer records, optionally filtered by name pattern."""

        params: list[Any] = [drawing_id]
        sql = """
            SELECT layer_name, flags, lineweight
            FROM drawing_layers
            WHERE drawing_id = %s
        """

        if layer_patterns:
            pattern_clauses = " OR ".join(
                "layer_name ILIKE %s" for _ in layer_patterns
            )
            sql += f" AND ({pattern_clauses})"
            for p in layer_patterns:
                params.append(p.replace("*", "%"))

        cur.execute(sql, params)
        rows = cur.fetchall()

        return [
            {
                "source":          "drawing_layers",
                "entity_handle":   None,
                "layer_name":      r["layer_name"],
                "flags":           r["flags"],
                "lineweight":      r["lineweight"],
                "similarity_score": None,
            }
            for r in rows
        ]

    # ── Stage 2: vector search ───────────────────────────────────── #

    def _vector_search(
        self,
        cur,
        drawing_id: int,
        query_text: str,
        top_k: int,
    ) -> list[dict]:
        """
        Embed query_text and find the most similar text chunks for this drawing.
        Returns empty list if embeddings are not available or query fails.
        """
        # Check if any embeddings exist for this drawing first
        cur.execute(
            """
            SELECT COUNT(*) AS n
            FROM drawing_text_chunks
            WHERE drawing_id = %s AND embedding IS NOT NULL
            """,
            (drawing_id,),
        )
        result = cur.fetchone()
        if not result or result["n"] == 0:
            log.debug(
                "No embeddings found for drawing_id=%d — skipping vector search",
                drawing_id,
            )
            return []

        # Embed the query text
        try:
            query_vector = self._embed_text(query_text)
        except Exception:
            log.exception("Failed to embed query text — skipping vector search")
            return []

        # pgvector cosine similarity search
        try:
            cur.execute(
                """
                SELECT
                    id,
                    entity_handle,
                    entity_type,
                    layer,
                    chunk_text,
                    position_json,
                    1 - (embedding <=> %s::vector) AS similarity_score
                FROM drawing_text_chunks
                WHERE drawing_id = %s
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (str(query_vector), drawing_id, str(query_vector), top_k),
            )
            rows = cur.fetchall()
        except Exception:
            log.exception("Vector search query failed — skipping")
            return []

        return [
            {
                "source":           "drawing_text_chunks",
                "entity_handle":    r["entity_handle"],
                "entity_type":      r["entity_type"],
                "layer":            r["layer"],
                "text":             r["chunk_text"],
                "position":         r["position_json"],
                "similarity_score": float(r["similarity_score"]),
            }
            for r in rows
        ]

    def _embed_text(self, text: str) -> list[float]:
        """Embed a single string using the configured embedding model."""
        try:
            import openai
        except ImportError as exc:
            raise RetrievalError(
                "openai package required for vector search: pip install openai"
            ) from exc

        kwargs: dict = {}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL

        client = openai.OpenAI(**kwargs)
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    # ── Deduplication + ranking ──────────────────────────────────── #

    def _deduplicate(self, evidence: list[dict]) -> list[dict]:
        """
        Remove duplicate evidence items by entity_handle.
        When the same handle appears from both metadata and vector search,
        keep the vector search version (it has a similarity_score).
        """
        seen: dict[str, dict] = {}

        for item in evidence:
            handle = item.get("entity_handle")

            if handle is None:
                # No handle (e.g. layer records, title block) — always keep
                seen[id(item)] = item  # type: ignore[index]
                continue

            if handle not in seen:
                seen[handle] = item
            else:
                # Prefer the item with a similarity score
                existing = seen[handle]
                if (
                    item.get("similarity_score") is not None
                    and existing.get("similarity_score") is None
                ):
                    seen[handle] = item

        return list(seen.values())

    def _rank(self, evidence: list[dict]) -> list[dict]:
        """
        Sort evidence by similarity_score descending.
        Items without a score (metadata-only) go to the end.
        """
        return sorted(
            evidence,
            key=lambda x: x.get("similarity_score") or -1,
            reverse=True,
        )