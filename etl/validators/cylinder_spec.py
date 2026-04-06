"""
cadsentinel.etl.validators.cylinder_spec
------------------------------------------
Validates that the required cylinder specification entries are present
in the drawing notes block.

Required entries (from JIT H Series Drawing Standard section 3):
    3.1  BORE   — cylinder bore diameter
    3.2  STROKE — cylinder stroke length
    3.3  ROD    — piston rod diameter
    3.4  PORTS  — port locations and sizes

Checks are performed against the combined text of all drawing text
chunks and entities. Matching is case-insensitive.

Returns:
    pass         — all four cylinder spec entries present
    fail         — one or more entries missing
    needs_review — no text evidence available
"""

from __future__ import annotations

from .base import (
    BaseValidator,
    ValidatorResult,
    collect_all_text,
    make_issue,
    make_evidence_ref,
    pass_result,
    fail_result,
    needs_review_result,
)

# Each entry is (spec_code, label, search_substring)
_REQUIRED_CYLINDER_SPECS: list[tuple[str, str, str]] = [
    ("3.1", "Bore",   "bore"),
    ("3.2", "Stroke", "stroke"),
    ("3.3", "Rod",    "rod"),
    ("3.4", "Ports",  "ports"),
]


class CylinderSpecValidator(BaseValidator):
    """
    Checks that BORE, STROKE, ROD, and PORTS entries exist in drawing notes.

    rule_config keys:
        severity_default (str): severity level, default 'high'
        required_specs (list):  optional override list of (code, label,
                                substring) tuples — uses JIT defaults
                                if not provided
    """

    name = "cylinder_spec"

    def _validate(
        self,
        evidence:    dict,
        rule_config: dict,
    ) -> ValidatorResult:

        severity = rule_config.get("severity_default", "high")

        # Collect and normalize all drawing text
        combined = collect_all_text(evidence)

        if not combined:
            return needs_review_result(
                reason   = (
                    "No text evidence available to check cylinder "
                    "specification entries."
                ),
                severity = severity,
            )

        # Allow rule_config to override required specs
        required = rule_config.get("required_specs", _REQUIRED_CYLINDER_SPECS)

        missing_issues = []
        found_specs    = []

        for spec_code, label, substring in required:
            if substring.lower() in combined:
                found_specs.append(make_evidence_ref(
                    source = "text_scan",
                    ref    = spec_code,
                    value  = label,
                ))
            else:
                missing_issues.append(make_issue(
                    issue_type    = "cylinder_spec_missing",
                    description   = (
                        f"Required cylinder specification '{label}' "
                        f"(rule {spec_code}) not found in drawing notes. "
                        f"Expected a NOTES block entry containing '{substring.upper()}'."
                    ),
                    severity      = severity,
                    suggested_fix = (
                        f"Add a {label.upper()} entry to the NOTES block. "
                        f"Example: {label.upper()} - [value]"
                    ),
                ))

        if missing_issues:
            missing_labels = [r[1] for r in required
                              if r[2].lower() not in combined]
            return fail_result(
                issue_summary = (
                    f"{len(missing_issues)} cylinder spec entry(s) missing: "
                    f"{', '.join(missing_labels)}."
                ),
                issues        = missing_issues,
                severity      = severity,
                evidence_used = found_specs,
            )

        return pass_result(
            severity      = severity,
            evidence_used = found_specs,
        )