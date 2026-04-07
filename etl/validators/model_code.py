"""
cadsentinel.etl.validators.model_code
---------------------------------------
Validates that a JIT H Series model code is present in the drawing.

Model code format:
    H-[Mount]-[Bore]X[Stroke]X[Rod]-[RodThread]-[Cushion]-[Port]-[Seal]-[Special]

Example:
    H-MT4-4X65X2-1-NC-S-V-S

Returns:
    pass         — model code found and parseable
    needs_review — no model code found (expected on component drawings)
    fail         — text matching model code pattern found but malformed
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

# JIT H Series model code pattern
# Handles both formats:
#   H-MT4-4X65X2-1-NC-S-V-S  (compact with X)
#   H-ME5-3.25 X 26.0 X 1.75 (spaced with X)
_MODEL_CODE_PATTERN = re.compile(
    r"\bH-[A-Z0-9]+-[\d.]+\s*[Xx]\s*[\d.]+\s*[Xx]\s*[\d.]+(?:-\S+)*",
    re.IGNORECASE,
)

# Partial match — something that looks like it's trying to be a model code
_PARTIAL_MODEL_PATTERN = re.compile(
    r"\bH-[A-Z0-9]+-[\d.]+",
    re.IGNORECASE,
)


class ModelCodeValidator(BaseValidator):
    """
    Checks for presence of a valid JIT H Series model code in drawing text.

    rule_config keys:
        severity_default (str): severity level, default 'high'
        model_code_attribute (str): optional title block attribute name to
                                    check first, default 'MATERIALS'
    """

    name = "model_code"

    def _validate(
        self,
        evidence:    dict,
        rule_config: dict,
    ) -> ValidatorResult:

        severity = rule_config.get("severity_default", "high")

        # ── Step 1: check title block attribute first ────────── #
        model_code = self._check_title_block(evidence, rule_config)

        # ── Step 2: fall back to full text scan ──────────────── #
        if not model_code:
            combined = collect_all_text(evidence)

            if not combined:
                return needs_review_result(
                    reason   = "No text evidence available to check for model code.",
                    severity = severity,
                )

            match = _MODEL_CODE_PATTERN.search(combined)
            if match:
                model_code = match.group(0).upper()
            else:
                # Check for partial match — malformed code present
                partial = _PARTIAL_MODEL_PATTERN.search(combined)
                if partial:
                    return fail_result(
                        issue_summary = "Partial or malformed JIT model code detected.",
                        issues        = [make_issue(
                            issue_type    = "model_code_malformed",
                            description   = (
                                f"Text '{partial.group(0)}' resembles a JIT model code "
                                f"but does not match the required format "
                                f"H-[Mount]-[Bore]X[Stroke]X[Rod]-[RodThread]-"
                                f"[Cushion]-[Port]-[Seal]-[Special]."
                            ),
                            severity      = severity,
                            suggested_fix = (
                                "Ensure the MATERIALS field contains the full "
                                "JIT H Series model code."
                            ),
                        )],
                        severity      = severity,
                        evidence_used = [],
                    )
                else:
                    # No model code at all — expected on component drawings
                    return needs_review_result(
                        reason   = (
                            "No JIT H Series model code found in drawing text. "
                            "This is expected for component drawings. "
                            "Verify manually if this is an assembly drawing."
                        ),
                        severity = severity,
                    )

        # ── Step 3: model code found — pass ──────────────────── #
        return pass_result(
            severity      = severity,
            evidence_used = [make_evidence_ref(
                source = "text_scan",
                ref    = "model_code",
                value  = model_code,
            )],
        )

    def _check_title_block(
        self,
        evidence:    dict,
        rule_config: dict,
    ) -> str | None:
        """
        Check title block attributes for model code.
        Returns the model code string if found, None otherwise.
        """
        attr_name = rule_config.get("model_code_attribute", "MATERIALS")

        for item in evidence.get("evidence", []):
            if item.get("source") != "drawing_title_block":
                continue
            attrs = item.get("attributes") or {}
            # Case-insensitive attribute lookup
            for key, val in attrs.items():
                if key.upper() == attr_name.upper() and val:
                    if _MODEL_CODE_PATTERN.search(str(val)):
                        return str(val).strip().upper()
        return None