# type_store.py

"""
cadsentinel.etl.classifiers.type_store
----------------------------------------
Persists and retrieves drawing type classifications.
"""

from __future__ import annotations

from .drawing_type_classifier import DrawingTypeResult
from ..db import get_connection


def save_drawing_type(
    drawing_id: int,
    result:     DrawingTypeResult,
    override:   bool = False,
) -> None:
    """Save a drawing type classification to the database."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('''
                SELECT id FROM drawing_types WHERE type_code = %s
            ''', (result.type_code,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Unknown drawing type: {result.type_code}")
            type_id = row["id"]

            cur.execute('''
                UPDATE drawings
                SET drawing_type_id         = %s,
                    drawing_type_confidence = %s,
                    drawing_type_source     = %s,
                    drawing_type_override   = %s
                WHERE id = %s
            ''', (type_id, result.confidence, result.source, override, drawing_id))
            conn.commit()


def get_drawing_type(drawing_id: int) -> DrawingTypeResult | None:
    """Retrieve the stored drawing type for a drawing."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('''
                SELECT dt.type_code, d.drawing_type_confidence,
                       d.drawing_type_source, d.drawing_type_override
                FROM drawings d
                JOIN drawing_types dt ON dt.id = d.drawing_type_id
                WHERE d.id = %s
            ''', (drawing_id,))
            row = cur.fetchone()
            if not row:
                return None
            return DrawingTypeResult(
                type_code  = row["type_code"],
                confidence = float(row["drawing_type_confidence"] or 0),
                source     = row["drawing_type_source"] or "unknown",
            )


def get_applicable_rule_ids(drawing_type_code: str) -> list[int]:
    """Get the rule IDs that apply to a given drawing type."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('''
                SELECT srdt.spec_rule_id
                FROM spec_rule_drawing_types srdt
                JOIN drawing_types dt ON dt.id = srdt.drawing_type_id
                WHERE dt.type_code = %s
                ORDER BY srdt.spec_rule_id
            ''', (drawing_type_code,))
            return [row["spec_rule_id"] for row in cur.fetchall()]
