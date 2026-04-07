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
    ("3.4", "Ports",  "port"),   # matches 'port', 'ports', 'nptf port' etc.
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

        combined = collect_all_text(evidence)

        if not combined:
            return needs_review_result(
                reason   = "No text evidence available to check cylinder specification entries.",
                severity = severity,
            )

        # If a specific entry is configured, check only that one
        single_spec = rule_config.get("required_spec")
        if single_spec:
            required = [tuple(single_spec)]
        else:
            required = rule_config.get("required_specs", _REQUIRED_CYLINDER_SPECS)

        missing_issues = []
        found_evidence = []

        for spec_code, label, substring in required:
            if substring.lower() in combined:
                found_evidence.append(make_evidence_ref(
                    source = "text_scan",
                    ref    = spec_code,
                    value  = label,
                ))
            else:
                missing_issues.append(make_issue(
                    issue_type    = "cylinder_spec_missing",
                    description   = (
                        f"{label} entry not found in drawing notes. "
                        f"Expected text containing: '{substring}'."
                    ),
                    severity      = severity,
                    suggested_fix = (
                        f"Add the {label} entry to the NOTES block. "
                        f"Example: {label.upper()} - [value]"
                    ),
                ))

        if missing_issues:
            missing_labels = [i["description"].split(" entry")[0] for i in missing_issues]
            return fail_result(
                issue_summary = (
                    f"{len(missing_issues)} cylinder spec entry(s) missing: "
                    f"{', '.join(missing_labels)}."
                ),
                issues        = missing_issues,
                severity      = severity,
                evidence_used = found_evidence,
            )

        return pass_result(
            severity      = severity,
            evidence_used = found_evidence,
        )