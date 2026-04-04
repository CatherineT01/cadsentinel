"""
cadsentinel.etl.validators.layer_naming
-----------------------------------------
Checks that drawing layer names conform to required patterns.

rule_config keys:
    required_patterns: list[str]  -- patterns that must appear (glob-style, * = wildcard)
                                     e.g. ["DIM*", "ANNO*", "BORDER"]
    forbidden_patterns: list[str] -- patterns that must NOT appear
    require_all: bool             -- True = ALL required_patterns must match at least one layer
                                     False = ANY required_pattern matching is a pass (default True)
    severity_default: str
"""

from __future__ import annotations

import fnmatch

from .base import (
    BaseValidator, ValidatorResult,
    pass_result, fail_result, needs_review_result, warning_result,
    make_issue, make_evidence_ref,
    get_layers,
)


class LayerNamingValidator(BaseValidator):
    """
    Deterministic validator for layer naming conventions.

    Checks:
      1. Required layer patterns — at least one layer must match each pattern
      2. Forbidden layer patterns — no layer may match these patterns
    """

    name = "layer_naming"

    def _validate(
        self,
        evidence:    dict,
        rule_config: dict,
    ) -> ValidatorResult:

        severity          = rule_config.get("severity_default", "medium")
        required_patterns = rule_config.get("required_patterns", [])
        forbidden_patterns = rule_config.get("forbidden_patterns", [])
        require_all       = rule_config.get("require_all", True)

        layers = get_layers(evidence)

        if not layers:
            return needs_review_result(
                reason   = "No layer evidence found in retrieved package. "
                           "Ensure 'layer' is included in the retrieval_recipe source_types.",
                severity = severity,
            )

        layer_names   = [l.get("layer_name", "") for l in layers if l.get("layer_name")]
        evidence_used = [
            make_evidence_ref("drawing_layers", name, name)
            for name in layer_names
        ]

        issues = []

        # ── Check required patterns ──────────────────────────── #
        unmatched_required = []
        for pattern in required_patterns:
            matched = _any_match(layer_names, pattern)
            if not matched:
                unmatched_required.append(pattern)

        if unmatched_required:
            if require_all or len(unmatched_required) == len(required_patterns):
                for pattern in unmatched_required:
                    issues.append(make_issue(
                        issue_type    = "required_layer_missing",
                        description   = f"No layer matching required pattern '{pattern}' found in drawing.",
                        severity      = severity,
                        suggested_fix = f"Add a layer named to match pattern '{pattern}'.",
                        entity_ref    = {"pattern": pattern},
                    ))

        # ── Check forbidden patterns ─────────────────────────── #
        for pattern in forbidden_patterns:
            matched_names = _all_matches(layer_names, pattern)
            for name in matched_names:
                issues.append(make_issue(
                    issue_type    = "forbidden_layer_present",
                    description   = f"Layer '{name}' matches forbidden pattern '{pattern}'.",
                    severity      = severity,
                    suggested_fix = f"Rename or remove layer '{name}'.",
                    entity_ref    = {"layer_name": name, "pattern": pattern},
                ))

        if not issues:
            return pass_result(severity=severity, evidence_used=evidence_used)

        required_issues  = [i for i in issues if i["issue_type"] == "required_layer_missing"]
        forbidden_issues = [i for i in issues if i["issue_type"] == "forbidden_layer_present"]

        parts = []
        if required_issues:
            patterns = [i["entity_ref"]["pattern"] for i in required_issues]
            parts.append(f"Missing required layers: {', '.join(patterns)}")
        if forbidden_issues:
            names = [i["entity_ref"]["layer_name"] for i in forbidden_issues]
            parts.append(f"Forbidden layers present: {', '.join(names)}")

        return fail_result(
            issue_summary = "; ".join(parts),
            issues        = issues,
            severity      = severity,
            evidence_used = evidence_used,
        )


# ── Pattern matching helpers ─────────────────────────────────── #

def _any_match(layer_names: list[str], pattern: str) -> bool:
    """Return True if any layer name matches the glob pattern (case-insensitive)."""
    pattern_upper = pattern.upper()
    return any(
        fnmatch.fnmatch(name.upper(), pattern_upper)
        for name in layer_names
    )


def _all_matches(layer_names: list[str], pattern: str) -> list[str]:
    """Return all layer names that match the glob pattern (case-insensitive)."""
    pattern_upper = pattern.upper()
    return [
        name for name in layer_names
        if fnmatch.fnmatch(name.upper(), pattern_upper)
    ]