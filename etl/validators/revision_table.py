"""
cadsentinel.etl.validators.revision_table
-------------------------------------------
Checks that a revision table exists in the drawing and contains
at least one revision entry.

rule_config keys:
    required_block_patterns: list[str]  -- block name patterns for revision tables
                                           default: ["REV*", "*REVISION*", "*REV_TABLE*"]
    min_revisions:           int        -- minimum number of revision entries required (default 1)
    severity_default:        str
"""

from __future__ import annotations

import fnmatch
from typing import Optional

from .base import (
    BaseValidator, ValidatorResult,
    pass_result, fail_result, needs_review_result, warning_result,
    make_issue, make_evidence_ref,
    get_entities, get_title_block,
)

DEFAULT_REV_PATTERNS = ["REV*", "*REVISION*", "*REV_TABLE*", "*REVISIONS*"]

# Common revision attribute tag names
REV_ATTR_TAGS = [
    "REV", "REVISION", "REV_NO", "REV_NUM",
    "CHANGE", "ECN", "ECO", "MOD",
]


class RevisionTableValidator(BaseValidator):
    """
    Deterministic validator for revision table existence.

    Detection strategy (in order):
    1. Look for INSERT entities whose block_name matches revision table patterns
    2. Look for revision-related attributes in the title block
    3. Look for ATTRIB entities with revision tag names
    """

    name = "revision_table"

    def _validate(
        self,
        evidence:    dict,
        rule_config: dict,
    ) -> ValidatorResult:

        severity         = rule_config.get("severity_default", "medium")
        block_patterns   = rule_config.get(
            "required_block_patterns", DEFAULT_REV_PATTERNS
        )
        min_revisions    = int(rule_config.get("min_revisions", 1))

        evidence_used = []
        issues        = []

        # ── Strategy 1: INSERT block name matching ────────────── #
        entities = get_entities(evidence)
        rev_inserts = []

        for ent in entities:
            block_name = ent.get("block_name") or ""
            if not block_name:
                continue
            if _matches_any_pattern(block_name, block_patterns):
                rev_inserts.append(ent)
                evidence_used.append(make_evidence_ref(
                    "drawing_entities",
                    ent.get("entity_handle", "insert"),
                    block_name,
                ))

        if rev_inserts:
            return pass_result(severity=severity, evidence_used=evidence_used)

        # ── Strategy 2: Title block revision attributes ───────── #
        tb = get_title_block(evidence)
        if tb:
            attrs = tb.get("attributes") or {}
            rev_attrs = _find_rev_attributes(attrs)
            if rev_attrs:
                evidence_used.append(make_evidence_ref(
                    "drawing_title_block",
                    tb.get("entity_handle", "title_block"),
                    rev_attrs,
                ))
                # Title block has revision info but no separate rev table
                # This is a warning, not a failure, if min_revisions == 1
                if min_revisions <= 1:
                    return pass_result(severity=severity, evidence_used=evidence_used)
                else:
                    return warning_result(
                        issue_summary = "Revision info found in title block only. "
                                        "A separate revision table block was not detected.",
                        issues        = [make_issue(
                            issue_type    = "revision_table_not_found",
                            description   = "Revision information exists in the title block "
                                            "but no dedicated revision table block was found.",
                            severity      = "low",
                            suggested_fix = "Add a dedicated revision table block to the drawing.",
                        )],
                        severity      = "low",
                        evidence_used = evidence_used,
                    )

        # ── Strategy 3: ATTRIB entities with revision tags ───── #
        rev_attribs = []
        for ent in entities:
            tag = (ent.get("tag") or "").upper()
            if any(tag.startswith(r) for r in [t.upper() for t in REV_ATTR_TAGS]):
                rev_attribs.append(ent)
                evidence_used.append(make_evidence_ref(
                    "drawing_entities",
                    ent.get("entity_handle", "attrib"),
                    tag,
                ))

        if rev_attribs:
            return pass_result(severity=severity, evidence_used=evidence_used)

        # ── No revision information found ─────────────────────── #

        # If no evidence was retrieved at all, flag for review
        if not evidence.get("evidence"):
            return needs_review_result(
                reason   = "No evidence was retrieved. Cannot check for revision table.",
                severity = severity,
            )

        return fail_result(
            issue_summary = "No revision table or revision information found in drawing.",
            issues        = [make_issue(
                issue_type    = "revision_table_missing",
                description   = "The drawing does not contain a detectable revision table. "
                                "A revision history is required for compliance.",
                severity      = severity,
                suggested_fix = "Add a revision table block to the drawing. "
                                f"Block name should match one of: "
                                f"{', '.join(block_patterns[:3])}",
            )],
            severity      = severity,
            confidence    = 0.90,
            evidence_used = evidence_used,
        )


# ── Helpers ──────────────────────────────────────────────────── #

def _matches_any_pattern(name: str, patterns: list[str]) -> bool:
    name_upper = name.upper()
    return any(
        fnmatch.fnmatch(name_upper, p.upper())
        for p in patterns
    )


def _find_rev_attributes(attributes: dict) -> dict:
    """Return any revision-related attributes from title block attrs."""
    found = {}
    for key, val in attributes.items():
        if any(r in key.upper() for r in ["REV", "REVISION", "ECN", "ECO"]):
            found[key] = val
    return found