# title_block.py

"""
cadsentinel.etl.validators.title_block
----------------------------------------
Checks that the drawing title block exists and contains
all required fields with non-empty values.

rule_config keys:
    required_fields: list[str]   -- attribute tag names that must be present
                                    e.g. ["DRAWING_NO", "REVISION", "TITLE"]
    severity_default: str        -- low | medium | high | critical
"""

from __future__ import annotations

from .base import (
    BaseValidator, ValidatorResult,
    pass_result, fail_result, needs_review_result,
    make_issue, make_evidence_ref,
    get_title_block,
)


class TitleBlockValidator(BaseValidator):
    """
    Deterministic validator for title block presence and field completeness.

    Passes if:
      - A title block was detected (found=True)
      - All required_fields are present in attributes
      - All required_fields have non-empty values

    Fails if:
      - No title block detected
      - One or more required fields are missing or empty
    """

    name = "title_block"

    def _validate(
        self,
        evidence:    dict,
        rule_config: dict,
    ) -> ValidatorResult:

        severity        = rule_config.get("severity_default", "high")
        required_fields = rule_config.get("required_fields", [])

        tb = get_title_block(evidence)

        # No title block in evidence at all
        if tb is None:
            return needs_review_result(
                reason   = "No title block evidence found in retrieved package. "
                           "Ensure title_block is included in the retrieval_recipe source_types.",
                severity = severity,
            )

        # Title block not detected in this drawing
        if not tb.get("found", False):
            return fail_result(
                issue_summary = "No title block detected in drawing.",
                issues        = [make_issue(
                    issue_type    = "title_block_missing",
                    description   = "The drawing does not contain a detectable title block. "
                                    "A title block is required for compliance.",
                    severity      = severity,
                    suggested_fix = "Add a title block block insert to the drawing.",
                )],
                severity      = severity,
                evidence_used = [make_evidence_ref("drawing_title_block", "title_block", False)],
            )

        # Title block found — check required fields
        attributes    = tb.get("attributes") or {}
        evidence_used = [make_evidence_ref(
            "drawing_title_block",
            tb.get("entity_handle", "title_block"),
            list(attributes.keys()),
        )]

        if not required_fields:
            # No specific fields required — presence alone is sufficient
            return pass_result(severity=severity, evidence_used=evidence_used)

        issues        = []
        missing       = []
        empty         = []

        for field_name in required_fields:
            # Case-insensitive lookup
            value = _get_attr(attributes, field_name)

            if value is None:
                missing.append(field_name)
                issues.append(make_issue(
                    issue_type    = "title_block_field_missing",
                    description   = f"Required title block field '{field_name}' is not present.",
                    severity      = severity,
                    suggested_fix = f"Add attribute '{field_name}' to the title block.",
                    entity_ref    = {"field": field_name},
                ))
            elif not str(value).strip():
                empty.append(field_name)
                issues.append(make_issue(
                    issue_type    = "title_block_field_empty",
                    description   = f"Required title block field '{field_name}' is present but empty.",
                    severity      = severity,
                    suggested_fix = f"Fill in a value for '{field_name}' in the title block.",
                    entity_ref    = {"field": field_name, "value": value},
                ))

        if not issues:
            return pass_result(severity=severity, evidence_used=evidence_used)

        parts = []
        if missing:
            parts.append(f"Missing fields: {', '.join(missing)}")
        if empty:
            parts.append(f"Empty fields: {', '.join(empty)}")

        return fail_result(
            issue_summary = "; ".join(parts),
            issues        = issues,
            severity      = severity,
            evidence_used = evidence_used,
        )


def _get_attr(attributes: dict, field_name: str):
    """Case-insensitive attribute lookup."""
    for key, value in attributes.items():
        if key.upper() == field_name.upper():
            return value
    return None