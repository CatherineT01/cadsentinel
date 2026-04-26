# ingestor.py

"""
cadsentinel.etl.ingestor
------------------------
Core ETL service: runs dwg_inspect, validates output, and orchestrates
all database insertions. Single entry point for drawing ingestion.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from .db import get_connection
from .parsers import (
    parse_layers,
    parse_blocks,
    parse_entities,
    parse_title_block,
    parse_dimensions,
    parse_text_chunks,
)

log = logging.getLogger(__name__)

SUPPORTED_SCHEMA_VERSIONS = {"1.1.0"}
DEFAULT_INSPECTOR_PATH = Path(__file__).parent.parent.parent / "dwg_inspect"


class IngestionError(RuntimeError):
    """Raised when ingestion cannot proceed."""


class DwgIngestor:
    """
    Orchestrates the full DWG -> PostgreSQL pipeline.

    Usage:
        ingestor = DwgIngestor(inspector_path="/path/to/dwg_inspect")
        drawing_id = ingestor.ingest("/path/to/drawing.dwg")
    """

    def __init__(
        self,
        inspector_path: str | Path = DEFAULT_INSPECTOR_PATH,
        timeout: int = 120,
    ):
        self.inspector_path = Path(inspector_path)
        self.timeout = timeout

        if not self.inspector_path.exists():
            raise IngestionError(
                f"dwg_inspect binary not found at: {self.inspector_path}\n"
                "Build it first: cd dwg_to_json && mkdir build && cd build "
                "&& cmake .. && cmake --build ."
            )

    # ── Public API ──────────────────────────────────────────────── #

    def ingest(self, dwg_path: str | Path, force: bool = False) -> int:
        """
        Full ingestion pipeline for one DWG file.
        Returns drawing_id (int) on success.
        Raises IngestionError on failure.
        """
        dwg_path = Path(dwg_path).resolve()
        if not dwg_path.exists():
            raise IngestionError(f"DWG file not found: {dwg_path}")

        log.info("Starting ingestion: %s", dwg_path)

        raw       = self._run_inspector(dwg_path)
        self._validate_schema(raw)
        file_hash = self._hash_file(dwg_path)

        with get_connection() as conn:
            with conn.cursor() as cur:

                existing = self._find_existing(cur, file_hash)
                if existing and not force:
                    log.info(
                        "Already ingested as drawing_id=%d (hash=%s). Skipping.",
                        existing, file_hash,
                    )
                    return existing
                elif existing and force:
                    log.info(
                        "Force re-ingesting: clearing hash for drawing_id=%d",
                        existing,
                    )
                    cur.execute(
                        "UPDATE drawings SET file_hash = NULL WHERE id = %s",
                        (existing,)
                    )
                    cur.execute(
                        "DELETE FROM drawing_versions WHERE file_hash = %s",
                        (file_hash,)
                    )
                    conn.commit()

                drawing_id = self._insert_drawing(cur, dwg_path, raw, file_hash)
                log.info("Inserted drawings row: drawing_id=%d", drawing_id)

                layer_count  = parse_layers(cur, drawing_id, raw.get("layers", []))
                log.info("Inserted %d layers", layer_count)

                block_count  = parse_blocks(cur, drawing_id, raw.get("blocks", []))
                log.info("Inserted %d blocks", block_count)

                entity_count = parse_entities(cur, drawing_id, raw.get("entities", []))
                log.info("Inserted %d entities", entity_count)

                parse_title_block(cur, drawing_id, raw.get("title_block", {}))
                log.info("Inserted title block")

                dim_count    = parse_dimensions(cur, drawing_id, raw.get("entities", []))
                log.info("Inserted %d dimensions", dim_count)

                chunk_count  = parse_text_chunks(cur, drawing_id, raw.get("entities", []))
                log.info("Inserted %d text chunks (pending embedding)", chunk_count)

                conn.commit()

        log.info("Ingestion complete: drawing_id=%d", drawing_id)
        return drawing_id

    # ── Private helpers ─────────────────────────────────────────── #

    def _run_inspector(self, dwg_path: Path) -> dict:
        log.debug("Running: %s %s", self.inspector_path, dwg_path)
        try:
            result = subprocess.run(
                [str(self.inspector_path), str(dwg_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise IngestionError(
                f"dwg_inspect timed out after {self.timeout}s for: {dwg_path}"
            ) from exc
        except FileNotFoundError as exc:
            raise IngestionError(
                f"dwg_inspect binary not executable: {self.inspector_path}"
            ) from exc

        if result.returncode != 0:
            raise IngestionError(
                f"dwg_inspect failed (exit {result.returncode}) for {dwg_path}:\n"
                f"{result.stderr.strip()}"
            )

        if not result.stdout.strip():
            raise IngestionError(f"dwg_inspect produced no output for: {dwg_path}")

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise IngestionError(
                f"dwg_inspect output was not valid JSON for {dwg_path}: {exc}"
            ) from exc

    def _validate_schema(self, raw: dict) -> None:
        version = raw.get("schema_version", "unknown")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise IngestionError(
                f"Unsupported dwg_inspect schema_version: '{version}'. "
                f"Expected one of: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )

    def _hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _find_existing(self, cur, file_hash: str) -> Optional[int]:
        cur.execute(
            "SELECT drawing_id FROM drawing_versions WHERE file_hash = %s LIMIT 1",
            (file_hash,),
        )
        row = cur.fetchone()
        return row["drawing_id"] if row else None

    def _insert_drawing(self, cur, dwg_path: Path, raw: dict, file_hash: str) -> int:
        header  = raw.get("header", {})
        summary = raw.get("summary", {})

        cur.execute(
            """
            INSERT INTO drawings (
                filename, original_path, dwg_version, codepage,
                schema_version, libredwg_version, num_entities,
                extents_json, raw_summary_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                dwg_path.name,
                str(dwg_path),
                header.get("version"),
                header.get("codepage"),
                raw.get("schema_version"),
                json.dumps(raw.get("libredwg_version", {})),
                summary.get("num_entities", 0),
                json.dumps(header.get("extents", {})),
                json.dumps(summary),
            ),
        )
        drawing_id: int = cur.fetchone()["id"]

        cur.execute(
            """
            INSERT INTO drawing_versions (drawing_id, version_num, file_hash, ingested_at)
            VALUES (%s, 1, %s, NOW())
            """,
            (drawing_id, file_hash),
        )
        return drawing_id