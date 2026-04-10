# spec_ingestor.py

"""
cadsentinel.etl.spec_ingestor
------------------------------
Orchestrates the full spec document ingestion pipeline:

    upload file
        → parse (PDF / DOCX / TXT)
        → chunk
        → LLM extract
        → normalize
        → store in DB (spec_documents, spec_document_sections, spec_rules)

Usage:
    ingestor = SpecIngestor(provider="openai")
    result   = ingestor.ingest("/path/to/spec.pdf")
    print(result["spec_document_id"], result["rules_extracted"])
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from .db import get_connection
from .spec_parser import parse_document, ParseError
from .spec_extractor import (
    get_llm_client,
    extract_specs_from_chunks,
    BaseLLMClient,
)
from .spec_models import NormalizedSpecRule, SpecDocument, SpecChunk

log = logging.getLogger(__name__)


class SpecIngestionError(RuntimeError):
    """Raised when spec ingestion cannot proceed."""


class SpecIngestor:
    """
    Full pipeline: file → parsed chunks → LLM extraction → database.

    Args:
        provider: "openai" or "grok"
        skip_extraction: If True, parse and store document/chunks but
                         skip LLM extraction. Useful for testing.
    """

    def __init__(
        self,
        provider: str = "openai",
        skip_extraction: bool = False,
    ):
        self.skip_extraction = skip_extraction
        if not skip_extraction:
            self.client: Optional[BaseLLMClient] = get_llm_client(provider)
        else:
            self.client = None

    # ── Public API ──────────────────────────────────────────────── #

    def ingest(self, file_path: str | Path) -> dict:
        """
        Full ingestion pipeline for one spec document.

        Returns:
            dict with keys: spec_document_id, chunks_stored,
                            rules_extracted, rules_stored
        """
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise SpecIngestionError(f"File not found: {file_path}")

        log.info("Starting spec ingestion: %s", file_path.name)

        # Step 1 — parse
        try:
            doc, chunks = parse_document(file_path)
        except ParseError as exc:
            raise SpecIngestionError(f"Parse failed: {exc}") from exc

        file_hash = self._hash_file(file_path)

        with get_connection() as conn:
            with conn.cursor() as cur:

                # Step 2 — check duplicate
                existing = self._find_existing(cur, file_hash)
                if existing:
                    log.info(
                        "Spec document already ingested as id=%d. Skipping.",
                        existing
                    )
                    return {
                        "spec_document_id": existing,
                        "chunks_stored":    0,
                        "rules_extracted":  0,
                        "rules_stored":     0,
                        "skipped":          True,
                    }

                # Step 3 — insert spec_documents
                spec_document_id = self._insert_spec_document(
                    cur, doc, file_hash, str(file_path)
                )
                log.info("Inserted spec_documents row: id=%d", spec_document_id)

                # Step 4 — insert spec_document_sections
                chunks_stored = self._insert_chunks(cur, spec_document_id, chunks)
                log.info("Inserted %d chunks", chunks_stored)

                conn.commit()

        # Step 5 — LLM extraction (outside transaction — can be slow)
        rules: list[NormalizedSpecRule] = []
        if not self.skip_extraction and self.client:
            rules = extract_specs_from_chunks(chunks, spec_document_id, self.client)

        # Step 6 — insert spec_rules
        rules_stored = 0
        if rules:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    rules_stored = self._insert_spec_rules(cur, rules)
                    log.info("Inserted %d spec_rules", rules_stored)
                    conn.commit()

        log.info(
            "Spec ingestion complete: doc_id=%d chunks=%d rules=%d",
            spec_document_id, chunks_stored, rules_stored
        )

        return {
            "spec_document_id": spec_document_id,
            "chunks_stored":    chunks_stored,
            "rules_extracted":  len(rules),
            "rules_stored":     rules_stored,
            "skipped":          False,
        }

    # ── Database helpers ─────────────────────────────────────────── #

    def _hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _find_existing(self, cur, file_hash: str) -> Optional[int]:
        cur.execute(
            "SELECT id FROM spec_documents WHERE file_hash = %s LIMIT 1",
            (file_hash,),
        )
        row = cur.fetchone()
        return row["id"] if row else None

    def _insert_spec_document(
        self,
        cur,
        doc: SpecDocument,
        file_hash: str,
        storage_path: str,
    ) -> int:
        cur.execute(
            """
            INSERT INTO spec_documents (
                filename, source_type, title, raw_text,
                file_hash, storage_path, parse_status
            ) VALUES (%s, %s, %s, %s, %s, %s, 'complete')
            RETURNING id
            """,
            (
                doc.filename,
                doc.source_type,
                doc.title,
                doc.raw_text,
                file_hash,
                storage_path,
            ),
        )
        return cur.fetchone()["id"]

    def _insert_chunks(
        self,
        cur,
        spec_document_id: int,
        chunks: list[SpecChunk],
    ) -> int:
        count = 0
        for chunk in chunks:
            try:
                cur.execute(
                    """
                    INSERT INTO spec_document_sections (
                        spec_document_id, chunk_index, raw_section_text,
                        char_start, char_end, source_page, section_hint
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        spec_document_id,
                        chunk.chunk_index,
                        chunk.raw_text,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.source_page,
                        chunk.section_hint,
                    ),
                )
                count += 1
            except Exception:
                log.exception(
                    "Failed to insert chunk %d for doc %d",
                    chunk.chunk_index, spec_document_id
                )
        return count

    def _insert_spec_rules(
        self,
        cur,
        rules: list[NormalizedSpecRule],
    ) -> int:
        count = 0
        for rule in rules:
            try:
                cur.execute(
                    """
                    INSERT INTO spec_rules (
                        spec_document_id,
                        spec_code,
                        spec_title,
                        original_spec_text,
                        normalized_rule_text,
                        rule_type,
                        execution_mode,
                        severity_default,
                        entity_scope,
                        expected_evidence_types,
                        checker_prompt,
                        retrieval_recipe,
                        structured_rule,
                        extraction_confidence,
                        normalization_confidence,
                        approved,
                        rule_version
                    ) VALUES (
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,FALSE,1
                    )
                    """,
                    (
                        rule.spec_document_id,
                        rule.spec_code,
                        rule.spec_title,
                        rule.original_spec_text,
                        rule.normalized_rule_text,
                        rule.rule_type,
                        rule.execution_mode,
                        rule.severity_default,
                        rule.entity_scope,
                        rule.expected_evidence_types,
                        rule.checker_prompt,
                        json.dumps(rule.retrieval_recipe),
                        json.dumps(rule.structured_rule),
                        rule.extraction_confidence,
                        rule.normalization_confidence,
                    ),
                )
                count += 1
            except Exception:
                log.exception(
                    "Failed to insert spec_rule '%s' for doc %d",
                    rule.spec_code, rule.spec_document_id
                )
        return count