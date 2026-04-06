"""
cadsentinel.etl.validators.jit_bore
-------------------------------------
Validates that the cylinder bore diameter is a valid JIT H Series bore size.

Valid JIT H Series bore sizes (inches):
    1.5, 2.0, 2.5, 3.25, 4.0, 5.0, 6.0, 7.0, 8.0

The bore value is extracted from drawing text by looking for a BORE entry
in the notes block. Falls back to scanning all text for the bore pattern.

Returns:
    pass         — bore value found and is a valid JIT bore size
    fail         — bore value found but not in valid list
    needs_review — no bore value found in drawing text
"""

from __future__ import annotations

import re

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

# Valid JIT H Series bore sizes in inches
_VALID_BORE_SIZES: frozenset[float] = frozenset({
    1.5, 2.0, 2.5, 3.25, 4.0, 5.0, 6.0, 7.0, 8.0
})

# Matches "BORE - 3.250" or "BORE: 3.25" or "BORE 4.0"
_BORE_PATTERN = re.compile(
    r"\bbore\s*[-:]\s*([\d.]+)",
    re.IGNORECASE,
)


class JITBoreValidator(BaseValidator):
    """
    Checks that the bore diameter is a valid JIT H Series bore size.

    rule_config keys:
        severity_default (str):    severity level, default 'high'
        valid_bore_sizes (list):   optional override list of valid bore
                                   sizes — uses JIT defaults if not provided
    """

    name = "jit_bore"

    def _validate(
        self,
        evidence:    dict,
        rule_config: dict,
    ) -> ValidatorResult:

        severity = rule_config.get("severity_default", "high")

        # Allow rule_config to override valid bore sizes
        valid_sizes = rule_config.get("valid_bore_sizes")
        if valid_sizes:
            valid_set = frozenset(float(v) for v in valid_sizes)
        else:
            valid_set = _VALID_BORE_SIZES

        # Collect and normalize all drawing text
        combined = collect_all_text(evidence)

        if not combined:
            return needs_review_result(
                reason   = "No text evidence available to check bore size.",
                severity = severity,
            )

        # Search for bore value in text
        match = _BORE_PATTERN.search(combined)

        if not match:
            return needs_review_result(
                reason   = (
                    "No BORE entry found in drawing text. "
                    "Expected a NOTES block entry in the format: BORE - [value]"
                ),
                severity = severity,
            )

        # Parse the bore value
        try:
            bore_value = float(match.group(1))
        except ValueError:
            return needs_review_result(
                reason   = (
                    f"Could not parse bore value from text: '{match.group(1)}'."
                ),
                severity = severity,
            )

        evidence_used = [make_evidence_ref(
            source = "text_scan",
            ref    = "bore",
            value  = bore_value,
        )]

        # Check against valid list
        if bore_value in valid_set:
            return pass_result(
                severity      = severity,
                evidence_used = evidence_used,
            )
        else:
            valid_sorted = sorted(valid_set)
            return fail_result(
                issue_summary = (
                    f"Bore size {bore_value}\" is not a valid "
                    f"JIT H Series bore size."
                ),
                issues        = [make_issue(
                    issue_type    = "invalid_bore_size",
                    description   = (
                        f"Bore diameter {bore_value}\" is not in the list of "
                        f"valid JIT H Series bore sizes: "
                        f"{valid_sorted}."
                    ),
                    severity      = severity,
                    suggested_fix = (
                        f"Use one of the valid JIT H Series bore sizes: "
                        f"{valid_sorted}."
                    ),
                )],
                severity      = severity,
                evidence_used = evidence_used,
            )