# parsers.py

"""
cadsentinel.etl.parsers
-----------------------
One parser function per drawing-side database table.
Each receives a psycopg2 cursor, a drawing_id, and the relevant
slice of dwg_inspect JSON. Returns a count of rows inserted.

Rules:
- Never raise on a single bad entity — log and skip, keep going.
- All geometry stored as JSONB.
- Structural DWG types (BLOCK, ENDBLK, SEQEND) are skipped.
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

_TEXT_CATEGORIES  = {"text"}
_STRUCTURAL_TYPES = {"BLOCK", "ENDBLK", "SEQEND", "UNUSED"}
_DIMENSION_CATEGORY = "dimension"
_MIN_CHUNK_LENGTH = 3


# ── Layers ────────────────────────────────────────────────────── #

def parse_layers(cur, drawing_id: int, layers: list[dict]) -> int:
    count = 0
    for layer in layers:
        name = layer.get("name") or ""
        if not name:
            continue
        try:
            cur.execute(
                """
                INSERT INTO drawing_layers (drawing_id, layer_name, flags, lineweight)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (drawing_id, layer_name) DO NOTHING
                """,
                (drawing_id, name, layer.get("flags", 0), layer.get("lineweight", 0)),
            )
            count += 1
        except Exception:
            log.exception("Failed to insert layer '%s' for drawing_id=%d", name, drawing_id)
    return count


# ── Blocks ────────────────────────────────────────────────────── #

def parse_blocks(cur, drawing_id: int, blocks: list[dict]) -> int:
    count = 0
    for block in blocks:
        name   = block.get("name") or ""
        handle = block.get("handle") or ""
        if not name:
            continue
        try:
            cur.execute(
                """
                INSERT INTO drawing_blocks
                    (drawing_id, block_name, handle, num_entities, entity_handles_json)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (drawing_id, handle) DO NOTHING
                """,
                (
                    drawing_id,
                    name,
                    handle,
                    block.get("num_entities", 0),
                    json.dumps(block.get("entity_handles", [])),
                ),
            )
            count += 1
        except Exception:
            log.exception("Failed to insert block '%s' for drawing_id=%d", name, drawing_id)
    return count


# ── Entities ──────────────────────────────────────────────────── #

def parse_entities(cur, drawing_id: int, entities: list[dict]) -> int:
    count = 0
    for ent in entities:
        etype = ent.get("type", "UNKNOWN")
        if etype in _STRUCTURAL_TYPES:
            continue
        try:
            cur.execute(
                """
                INSERT INTO drawing_entities
                    (drawing_id, entity_type, category, layer, handle,
                     owner_handle, dwg_index, text_content, geometry_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    drawing_id,
                    etype,
                    ent.get("category", "other"),
                    ent.get("layer"),
                    ent.get("handle"),
                    ent.get("owner_handle"),
                    ent.get("index"),
                    ent.get("text"),
                    json.dumps(ent.get("geometry")) if ent.get("geometry") else None,
                ),
            )
            count += 1
        except Exception:
            log.exception(
                "Failed to insert entity type='%s' handle='%s' drawing_id=%d",
                etype, ent.get("handle"), drawing_id,
            )
    return count


# ── Title block ───────────────────────────────────────────────── #

def parse_title_block(cur, drawing_id: int, title_block: dict) -> None:
    found      = title_block.get("found", False)
    attributes = title_block.get("attributes", {})

    # Filename fallback for drawing number
    if found and not attributes.get("DWG NUMBER"):
        cur.execute("SELECT filename FROM drawings WHERE id = %s", (drawing_id,))
        row = cur.fetchone()
        if row and row["filename"]:
            import os
            fname = os.path.basename(row["filename"])
            # Strip extension and revision suffix (REV A, REV B, etc.)
            import re
            name = re.sub(r'\.(dwg|dxf)$', '', fname, flags=re.IGNORECASE)
            name = re.sub(r'\s+REV\s+\w+$', '', name, flags=re.IGNORECASE).strip()
            attributes = dict(attributes)
            attributes["DWG NUMBER"] = name

    if not found:
        confidence = 0.0
    elif len(attributes) > 4:
        confidence = 0.90
    elif len(attributes) > 1:
        confidence = 0.75
    elif found:
        confidence = 0.50
    else:
        confidence = 0.0

    try:
        cur.execute(
            """
            INSERT INTO drawing_title_block
                (drawing_id, found, block_name, handle, layer,
                 attributes_json, geometry_json, candidates_json, detection_confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (drawing_id) DO UPDATE SET
                found                = EXCLUDED.found,
                block_name           = EXCLUDED.block_name,
                handle               = EXCLUDED.handle,
                layer                = EXCLUDED.layer,
                attributes_json      = EXCLUDED.attributes_json,
                geometry_json        = EXCLUDED.geometry_json,
                candidates_json      = EXCLUDED.candidates_json,
                detection_confidence = EXCLUDED.detection_confidence,
                updated_at           = NOW()
            """,
            (
                drawing_id,
                found,
                title_block.get("block_name"),
                title_block.get("handle"),
                title_block.get("layer"),
                json.dumps(attributes),
                json.dumps(title_block.get("geometry", {})),
                json.dumps(title_block.get("candidates", [])),
                confidence,
            ),
        )
    except Exception:
        log.exception("Failed to insert title block for drawing_id=%d", drawing_id)


# ── Dimensions ────────────────────────────────────────────────── #

def parse_dimensions(cur, drawing_id: int, entities: list[dict]) -> int:
    count = 0
    for ent in entities:
        if ent.get("category") != _DIMENSION_CATEGORY:
            continue
        geometry = ent.get("geometry", {})
        try:
            cur.execute(
                """
                INSERT INTO drawing_dimensions
                    (drawing_id, dim_type, handle, layer,
                     measured_value, user_text, text_position_json, geometry_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    drawing_id,
                    ent.get("type", "UNKNOWN"),
                    ent.get("handle"),
                    ent.get("layer"),
                    ent.get("value"),
                    ent.get("text"),
                    json.dumps(geometry.get("text_position")) if geometry.get("text_position") else None,
                    json.dumps(geometry),
                ),
            )
            count += 1
        except Exception:
            log.exception(
                "Failed to insert dimension handle='%s' drawing_id=%d",
                ent.get("handle"), drawing_id,
            )
    return count


# ── Text chunks ───────────────────────────────────────────────── #

def parse_text_chunks(cur, drawing_id: int, entities: list[dict]) -> int:
    count = 0
    for ent in entities:
        if ent.get("category") not in _TEXT_CATEGORIES:
            continue

        text_content = (ent.get("text") or "").strip()
        if len(text_content) < _MIN_CHUNK_LENGTH:
            continue

        geometry = ent.get("geometry", {})
        ins_pt   = geometry.get("ins_pt") if geometry else None

        try:
            cur.execute(
                """
                INSERT INTO drawing_text_chunks
                    (drawing_id, entity_handle, entity_type, layer,
                     chunk_text, position_json)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    drawing_id,
                    ent.get("handle"),
                    ent.get("type", "UNKNOWN"),
                    ent.get("layer"),
                    text_content,
                    json.dumps(ins_pt) if ins_pt else None,
                ),
            )
            count += 1
        except Exception:
            log.exception(
                "Failed to insert text chunk handle='%s' drawing_id=%d",
                ent.get("handle"), drawing_id,
            )
    return count


# ── Utility ───────────────────────────────────────────────────── #

def extract_text_from_entity(ent: dict[str, Any]) -> str:
    """Return the best text string from any entity dict."""
    direct = (ent.get("text") or "").strip()
    if direct:
        return direct
    default = (ent.get("default_value") or "").strip()
    if default:
        return default
    return ""