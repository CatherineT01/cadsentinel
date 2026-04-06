"""
cadsentinel.etl.validators.jit_mount
--------------------------------------
Validates that the mounting style code in the JIT H Series model code
is a valid JIT mounting style.

Valid JIT H Series mounting style codes (24 total):
    MHF, MCF, ME5, ME6, MF1, MF2, MF5, MF6, MP1, MP2, MP3, MPU3,
    MS1, MS2, MS3, MS4, MS7, MT1, MT2, MT4, MX0, MX1, MX2, MX3

The mount code is extracted from the model code found in drawing text.
Model code format: H-[MOUNT]-[BORE]X[STROKE]X[ROD]-...

Returns:
    pass         — mount code found and is a valid JIT mounting style
    fail         — mount code found but not in valid list
    needs_review — no model code found in drawing text
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

# Valid JIT H Series mounting style codes
_VALID_MOUNT_CODES: frozenset[str] = frozenset({
    "MHF", "MCF", "ME5", "ME6", "MF1", "MF2", "MF5", "MF6",
    "MP1", "MP2", "MP3", "MPU3",
    "MS1", "MS2", "MS3", "MS4", "MS7",
    "MT1", "MT2", "MT4",
    "MX0", "MX1", "MX2", "MX3",
})

# Full model code pattern — captures mount code in group 1
_MODEL_CODE_PATTERN = re.compile(
    r"\bH-([A-Z0-9]+)-[\d.]+X[\d.]+X[\d.]+-\S+-\S+-\S+-\S+-\S+",
    re.IGNORECASE,
)

# Partial pattern — captures mount code even if rest of code is incomplete
_PARTIAL_MOUNT_PATTERN = re.compile(
    r"\bH-([A-Z0-9]+)-[\d.]+",
    re.IGNORECASE,
)


class JITMountValidator(BaseValidator):
    """
    Checks that the mounting style code is a valid JIT H Series mount code.

    rule_config keys:
        severity_default (str):    severity level, default 'high'
        valid_mount_codes (list):  optional override list of valid mount
                                   codes — uses JIT defaults if not provided
    """

    name = "jit_mount"

    def _validate(
        self,
        evidence:    dict,
        rule_config: dict,
    ) -> ValidatorResult:

        severity = rule_config.get("severity_default", "high")

        # Allow rule_config to override valid mount codes
        valid_codes = rule_config.get("valid_mount_codes")
        if valid_codes:
            valid_set = frozenset(c.upper() for c in valid_codes)
        else:
            valid_set = _VALID_MOUNT_CODES

        # Collect and normalize all drawing text
        combined = collect_all_text(evidence)

        if not combined:
            return needs_review_result(
                reason   = "No text evidence available to check mount code.",
                severity = severity,
            )

        # Try full model code match first
        match      = _MODEL_CODE_PATTERN.search(combined)
        is_partial = False

        if not match:
            # Try partial match
            match      = _PARTIAL_MOUNT_PATTERN.search(combined)
            is_partial = True

        if not match:
            return needs_review_result(
                reason   = (
                    "No JIT H Series model code found in drawing text. "
                    "Cannot validate mounting style. "
                    "This is expected for component drawings."
                ),
                severity = severity,
            )

        # Extract mount code
        mount_code = match.group(1).upper()

        evidence_used = [make_evidence_ref(
            source = "text_scan",
            ref    = "mount_code",
            value  = mount_code,
        )]

        # Check against valid list
        if mount_code in valid_set:
            return pass_result(
                severity      = severity,
                evidence_used = evidence_used,
            )
        else:
            valid_sorted = sorted(valid_set)
            partial_note = (
                " (extracted from partial model code)" if is_partial else ""
            )
            return fail_result(
                issue_summary = (
                    f"Mount code '{mount_code}' is not a valid "
                    f"JIT H Series mounting style."
                ),
                issues        = [make_issue(
                    issue_type    = "invalid_mount_code",
                    description   = (
                        f"Mount code '{mount_code}'{partial_note} is not in "
                        f"the list of valid JIT H Series mounting styles: "
                        f"{valid_sorted}."
                    ),
                    severity      = severity,
                    suggested_fix = (
                        f"Use one of the valid JIT H Series mount codes: "
                        f"{valid_sorted}."
                    ),
                )],
                severity      = severity,
                evidence_used = evidence_used,
            )