# confidentiality.py

"""
cadsentinel.etl.validators.confidentiality
--------------------------------------------
Validates that the drawing contains the JIT Industries proprietary
and confidential statement.

Required text (partial match, case-insensitive):
    "proprietary and confidential"

Additional keywords checked as fallback for older drawings:
    "sole property of jit"
    "jit industries is strictly prohibited"

Returns:
    pass         -- confidentiality statement found
    fail         -- no confidentiality statement found
    needs_review -- no text evidence available
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

# Primary match
_PRIMARY = "proprietary and confidential"

# Fallback matches for older drawings with different wording
_FALLBACKS = [
    "sole property of jit",
    "strictly prohibited",
    "written permission of jit",
]


class ConfidentialityValidator(BaseValidator):
    """
    Checks that the drawing contains the JIT Industries
    proprietary and confidential statement.
    """

    name = "confidentiality"

    def _validate(
        self,
        evidence:    dict,
        rule_config: dict,
    ) -> ValidatorResult:

        severity = rule_config.get("severity_default", "high")

        combined = collect_all_text(evidence)

        if not combined:
            return needs_review_result(
                reason   = "No text evidence available to check confidentiality statement.",
                severity = severity,
            )

        # Primary check
        if _PRIMARY in combined:
            return pass_result(
                severity      = severity,
                evidence_used = [make_evidence_ref(
                    source = "text_scan",
                    ref    = "confidentiality",
                    value  = _PRIMARY,
                )],
            )

        # Fallback check for older drawings
        for fallback in _FALLBACKS:
            if fallback in combined:
                return pass_result(
                    severity      = severity,
                    evidence_used = [make_evidence_ref(
                        source = "text_scan",
                        ref    = "confidentiality_legacy",
                        value  = fallback,
                    )],
                )

        return fail_result(
            issue_summary = "Proprietary and confidential statement not found in drawing.",
            issues        = [make_issue(
                issue_type    = "confidentiality_missing",
                description   = (
                    "The JIT Industries proprietary and confidential statement "
                    "was not found in the drawing text. Required text: "
                    "'PROPRIETARY AND CONFIDENTIAL -- The information contained "
                    "in this drawing is the sole property of JIT Industries.'"
                ),
                severity      = severity,
                suggested_fix = (
                    "Add the JIT Industries proprietary and confidential "
                    "statement to the drawing title block."
                ),
            )],
            severity      = severity,
            evidence_used = [],
        )