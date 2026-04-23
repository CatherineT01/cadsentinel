"""
cadsentinel.etl.compatibility.spec_extractor
----------------------------------------------
Extracts key dimensional specs from ingested drawings for
compatibility checking.

Extracts:
    - bore diameter
    - stroke length
    - rod diameter
    - port sizes
    - model code
    - drawing type
    - key dimensions from drawing_entities
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_BORE_PATTERN   = re.compile(r"bore\s*[-:]?\s*([\d.]+)", re.IGNORECASE)
_STROKE_PATTERN = re.compile(r"stroke\s*[-:]?\s*([\d.]+)", re.IGNORECASE)
_ROD_PATTERN    = re.compile(r"rod\s*(?:dia\.?)?\s*[-:]?\s*([\d.]+)", re.IGNORECASE)
_PORT_PATTERN   = re.compile(r"([\d.]+)\s*(?:npt|nptf|sae|port)", re.IGNORECASE)
_MODEL_PATTERN  = re.compile(
    r"\bH-[A-Z0-9]+-(\d+\.?\d*)\s*[Xx]\s*([\d.]+)\s*[Xx]\s*([\d.]+)",
    re.IGNORECASE,
)


@dataclass
class DrawingSpecs:
    drawing_id:   int
    filename:     str
    drawing_type: str
    bore:         float | None = None
    stroke:       float | None = None
    rod:          float | None = None
    ports:        list[float]  = field(default_factory=list)
    model_code:   str | None   = None
    raw_text:     str          = ""


def extract_specs(drawing_id: int) -> DrawingSpecs:
    """Extract key specs from a drawing in the database."""
    from ..db import get_connection
    from ..validators.base import collect_all_text, _strip_mtext

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Get drawing info
            cur.execute("""
                SELECT d.filename, dt.type_code
                FROM drawings d
                LEFT JOIN drawing_types dt ON dt.id = d.drawing_type_id
                WHERE d.id = %s
            """, (drawing_id,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Drawing {drawing_id} not found")

            filename     = row["filename"] or ""
            drawing_type = row["type_code"] or "unknown"

            # Get all text content
            cur.execute("""
                SELECT text_content
                FROM drawing_entities
                WHERE drawing_id = %s
                  AND category = 'text'
                  AND text_content IS NOT NULL
                ORDER BY LENGTH(text_content) DESC
                LIMIT 100
            """, (drawing_id,))
            texts = [_strip_mtext(r["text_content"]) for r in cur.fetchall()]
            combined = " ".join(texts).lower()

    specs = DrawingSpecs(
        drawing_id   = drawing_id,
        filename     = filename,
        drawing_type = drawing_type,
        raw_text     = combined,
    )

    # Extract model code first — gives bore/stroke/rod directly
    model_match = _MODEL_PATTERN.search(combined)
    if model_match:
        specs.model_code = model_match.group(0)
        try:
            specs.bore   = float(model_match.group(1))
            specs.stroke = float(model_match.group(2))
            specs.rod    = float(model_match.group(3))
        except ValueError:
            pass

    # Extract bore from notes if not found in model code
    if specs.bore is None:
        m = _BORE_PATTERN.search(combined)
        if m:
            try:
                specs.bore = float(m.group(1))
            except ValueError:
                pass

    # Extract stroke
    if specs.stroke is None:
        m = _STROKE_PATTERN.search(combined)
        if m:
            try:
                specs.stroke = float(m.group(1))
            except ValueError:
                pass

    # Extract rod diameter
    if specs.rod is None:
        m = _ROD_PATTERN.search(combined)
        if m:
            try:
                specs.rod = float(m.group(1))
            except ValueError:
                pass

    # Extract port sizes
    for m in _PORT_PATTERN.finditer(combined):
        try:
            specs.ports.append(float(m.group(1)))
        except ValueError:
            pass

    return specs


def extract_specs_batch(drawing_ids: list[int]) -> list[DrawingSpecs]:
    """Extract specs from multiple drawings."""
    results = []
    for drawing_id in drawing_ids:
        try:
            specs = extract_specs(drawing_id)
            results.append(specs)
            log.debug(
                f"id={drawing_id} type={specs.drawing_type} "
                f"bore={specs.bore} stroke={specs.stroke} rod={specs.rod}"
            )
        except Exception as e:
            log.warning(f"Failed to extract specs for drawing {drawing_id}: {e}")
    return results