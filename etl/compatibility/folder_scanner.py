"""
cadsentinel.etl.compatibility.folder_scanner
----------------------------------------------
Scans a folder for DWG files and ingests any that are new or modified.

Duplicate detection:
    - SHA-256 hash of file contents
    - If hash matches existing record: skip ingestion, return existing drawing_id
    - If filename matches but hash differs: re-ingest as new version
    - If neither match: ingest as brand new drawing

Returns a list of ScanResult objects with drawing_id, filename, and status.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

DWG_EXTENSIONS = {".dwg"}


@dataclass
class ScanResult:
    filename:    str
    filepath:    str
    drawing_id:  int | None
    status:      str        # 'new', 'existing', 'updated', 'failed'
    file_hash:   str = ""
    file_version: int = 1
    error:       str = ""


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def scan_folder(
    folder_path:     str,
    inspector_path:  str,
    provider:        str = "openai",
    force_reingest:  bool = False,
) -> list[ScanResult]:
    """
    Scan a folder for DWG files and ingest new or modified ones.

    Args:
        folder_path:    Path to folder containing DWG files
        inspector_path: Path to dwg_inspect.exe
        provider:       LLM provider for drawing type classification
        force_reingest: If True, re-ingest all drawings even if unchanged

    Returns:
        List of ScanResult objects
    """
    from ..db import get_connection
    from ..ingestor import DwgIngestor
    from ..classifiers.drawing_type_classifier import DrawingTypeClassifier
    from ..classifiers.type_store import save_drawing_type

    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")

    # Find all DWG files
    dwg_files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in DWG_EXTENSIONS
    ]

    if not dwg_files:
        log.warning(f"No DWG files found in {folder_path}")
        return []

    log.info(f"Found {len(dwg_files)} DWG file(s) in {folder_path}")

    ingestor   = DwgIngestor(inspector_path=inspector_path)
    classifier = DrawingTypeClassifier(provider=provider)
    results    = []

    for dwg_file in sorted(dwg_files):
        result = _process_file(
            dwg_file       = dwg_file,
            ingestor       = ingestor,
            classifier     = classifier,
            force_reingest = force_reingest,
        )
        results.append(result)
        log.info(
            f"  {result.status.upper():<10} id={result.drawing_id}  {result.filename}"
        )

    return results


def _process_file(
    dwg_file:       Path,
    ingestor:       object,
    classifier:     object,
    force_reingest: bool,
) -> ScanResult:
    """Process a single DWG file — check for duplicates, ingest if needed."""
    from ..db import get_connection
    from ..classifiers.type_store import save_drawing_type
    from ..retriever import EvidenceRetriever

    filepath = str(dwg_file)
    filename = dwg_file.name

    try:
        # Compute hash
        file_hash     = compute_file_hash(filepath)
        file_size     = dwg_file.stat().st_size

        # Check DB for existing record
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check by hash first
                cur.execute(
                    "SELECT id, filename, file_version FROM drawings WHERE file_hash = %s",
                    (file_hash,)
                )
                existing_by_hash = cur.fetchone()

                if existing_by_hash and not force_reingest:
                    # Exact match — skip ingestion
                    return ScanResult(
                        filename    = filename,
                        filepath    = filepath,
                        drawing_id  = existing_by_hash["id"],
                        status      = "existing",
                        file_hash   = file_hash,
                        file_version = existing_by_hash["file_version"],
                    )

                # Check by filename for version tracking
                cur.execute(
                    "SELECT id, file_version, file_hash FROM drawings WHERE filename = %s ORDER BY file_version DESC LIMIT 1",
                    (filename,)
                )
                existing_by_name = cur.fetchone()

        # Determine version number
        if existing_by_name and existing_by_name["file_hash"] != file_hash:
            new_version = existing_by_name["file_version"] + 1
            status      = "updated"
        else:
            new_version = 1
            status      = "new"

        # Ingest the drawing
        drawing_id = ingestor.ingest(filepath)

        # Update file metadata
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE drawings
                       SET file_hash        = %s,
                           file_version     = %s,
                           file_size_bytes  = %s,
                           original_path    = %s
                       WHERE id = %s""",
                    (file_hash, new_version, file_size, filepath, drawing_id)
                )
                conn.commit()

        # Classify drawing type
        retriever = EvidenceRetriever(embed_for_vector_search=False)
        package   = retriever.retrieve(
            drawing_id           = drawing_id,
            spec_rule_id         = 1,
            normalized_rule_text = "drawing type classification",
            retrieval_recipe     = {
                "source_types": ["note", "entity"],
                "top_k": 30,
                "keyword_filters": [],
                "entity_filters": {}
            },
        )
        type_result = classifier.classify(
            drawing_id = drawing_id,
            filename   = filename,
            evidence   = package,
            dwg_path   = filepath,
        )
        save_drawing_type(drawing_id, type_result)

        return ScanResult(
            filename     = filename,
            filepath     = filepath,
            drawing_id   = drawing_id,
            status       = status,
            file_hash    = file_hash,
            file_version = new_version,
        )

    except Exception as e:
        log.exception(f"Failed to process {filename}: {e}")
        return ScanResult(
            filename   = filename,
            filepath   = filepath,
            drawing_id = None,
            status     = "failed",
            error      = str(e),
        )