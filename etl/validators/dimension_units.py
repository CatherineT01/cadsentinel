# dimension_units.py

"""
cadsentinel.etl.validators.dimension_units
-------------------------------------------
Checks that drawing dimensions use the required unit system.

rule_config keys:
    required_unit:   str   -- "inches" | "mm" | "feet" | "meters"
    unit_keywords:   list[str]  -- text strings indicating the required unit
                                   defaults to ["in", "inch", "inches", "\""] for inches
                                   or ["mm", "millimeter"] for mm
    allow_mixed:     bool  -- if True, warn instead of fail on mixed units (default False)
    severity_default: str
"""

from __future__ import annotations

import re
from typing import Optional

from .base import (
    BaseValidator, ValidatorResult,
    pass_result, fail_result, needs_review_result, warning_result,
    make_issue, make_evidence_ref,
    get_dimensions, get_title_block, get_text_chunks, get_entities,
)

# Unit keyword sets
UNIT_KEYWORDS: dict[str, list[str]] = {
    "inches": ["inches", "inch", " in", '"', "in."],
    "mm":     ["mm", "millimeter", "millimeters", "millimetre"],
    "feet":   ["feet", "foot", "ft", "'"],
    "meters": ["meters", "metre", "metres", " m"],
}

# Unit keywords that indicate the WRONG unit for each required unit
CONFLICTING_KEYWORDS: dict[str, list[str]] = {
    "inches": ["mm", "millimeter", "millimeters"],
    "mm":     ["inches", "inch", '"'],
    "feet":   ["mm", "meters", "metre"],
    "meters": ["feet", "foot", "inches", "inch"],
}


class DimensionUnitsValidator(BaseValidator):
    """
    Deterministic validator for dimension unit consistency.

    Strategy:
    1. Check title block for unit declaration (most authoritative)
    2. Check dimension user_text fields for unit indicators
    3. Check general notes for unit declarations (e.g. "ALL DIMENSIONS IN INCHES")
    4. Flag any conflicting unit indicators as failures
    """

    name = "dimension_units"

    def _validate(
        self,
        evidence:    dict,
        rule_config: dict,
    ) -> ValidatorResult:

        severity      = rule_config.get("severity_default", "medium")
        required_unit = rule_config.get("required_unit", "inches").lower()
        allow_mixed   = rule_config.get("allow_mixed", False)

        required_keywords   = UNIT_KEYWORDS.get(required_unit, [required_unit])
        conflicting_keywords = CONFLICTING_KEYWORDS.get(required_unit, [])

        evidence_used  = []
        issues         = []
        unit_confirmed = False
        conflicts      = []

        # ── 1. Title block unit check ────────────────────────── #
        tb = get_title_block(evidence)
        if tb:
            attrs = tb.get("attributes") or {}
            unit_attr = _get_unit_attr(attrs)
            if unit_attr:
                evidence_used.append(make_evidence_ref(
                    "drawing_title_block", "UNITS", unit_attr
                ))
                if _text_matches_unit(unit_attr, required_keywords):
                    unit_confirmed = True
                elif _text_matches_any(unit_attr, conflicting_keywords):
                    conflicts.append(f"Title block UNITS field states '{unit_attr}'")

        # ── 2. Dimension user_text check ─────────────────────── #
        dimensions = get_dimensions(evidence)
        for dim in dimensions:
            user_text = dim.get("user_text") or ""
            if not user_text.strip():
                continue

            evidence_used.append(make_evidence_ref(
                "drawing_dimensions",
                dim.get("entity_handle", "dim"),
                user_text,
            ))

            if _text_matches_any(user_text, conflicting_keywords):
                conflicts.append(
                    f"Dimension '{user_text}' appears to use wrong units"
                )

        # ── 3. Note / text chunk check ───────────────────────── #
        text_items = get_text_chunks(evidence) + get_entities(evidence)
        for item in text_items:
            text = item.get("text") or item.get("chunk_text") or ""
            if not text.strip():
                continue

            text_upper = text.upper()

            # Look for unit declaration notes
            if _is_unit_declaration(text_upper):
                evidence_used.append(make_evidence_ref(
                    item.get("source", "text"),
                    item.get("entity_handle", "note"),
                    text[:120],
                ))
                if _text_matches_unit(text_upper, [u.upper() for u in required_keywords]):
                    unit_confirmed = True
                elif _text_matches_any(text_upper, [c.upper() for c in conflicting_keywords]):
                    conflicts.append(f"Note states conflicting units: '{text[:80]}'")

        # ── Build result ─────────────────────────────────────── #

        # Conflicting units → always fail (or warn if allow_mixed)
        if conflicts:
            for conflict in conflicts:
                issues.append(make_issue(
                    issue_type    = "unit_conflict",
                    description   = conflict,
                    severity      = severity,
                    suggested_fix = f"Ensure all dimensions use {required_unit}. "
                                    "Check title block, dimension text, and general notes.",
                ))

            if allow_mixed:
                return warning_result(
                    issue_summary = f"Mixed units detected. Required: {required_unit}.",
                    issues        = issues,
                    severity      = "low",
                    evidence_used = evidence_used,
                )
            return fail_result(
                issue_summary = f"Unit conflicts detected. Required: {required_unit}. "
                                f"Found: {'; '.join(conflicts)}",
                issues        = issues,
                severity      = severity,
                evidence_used = evidence_used,
            )

        # No conflicting units found
        if unit_confirmed:
            return pass_result(severity=severity, evidence_used=evidence_used)

        # No unit information found at all
        if not evidence_used:
            return needs_review_result(
                reason   = f"No unit information found in evidence. "
                           "Cannot confirm whether dimensions use {required_unit}.",
                severity = severity,
            )

        # Evidence found but unit not explicitly confirmed — soft pass with warning
        return warning_result(
            issue_summary = f"Unit '{required_unit}' not explicitly declared in drawing. "
                            "No conflicting units detected.",
            issues        = [make_issue(
                issue_type    = "unit_not_declared",
                description   = f"The required unit ({required_unit}) is not explicitly declared "
                                "in the title block or general notes.",
                severity      = "low",
                suggested_fix = f"Add a general note stating 'ALL DIMENSIONS IN "
                                f"{required_unit.upper()} UNLESS OTHERWISE NOTED'.",
            )],
            severity      = "low",
            confidence    = 0.75,
            evidence_used = evidence_used,
        )


# ── Text matching helpers ────────────────────────────────────── #

def _text_matches_unit(text: str, keywords: list[str]) -> bool:
    """Return True if text contains any of the unit keywords."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _text_matches_any(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _is_unit_declaration(text_upper: str) -> bool:
    """Heuristic: does this text look like a unit declaration note?"""
    indicators = [
        "DIMENSION", "UNIT", "UNLESS OTHERWISE", "ALL DIM",
        "TOLERAN", "SCALE",
    ]
    return any(ind in text_upper for ind in indicators)


def _get_unit_attr(attributes: dict) -> Optional[str]:
    """Find a unit-related attribute in title block attributes."""
    unit_keys = ["UNITS", "UNIT", "DIM_UNITS", "DRAWING_UNITS", "MEASUREMENT"]
    for key in unit_keys:
        for attr_key, val in attributes.items():
            if attr_key.upper() == key:
                return str(val)
    return None