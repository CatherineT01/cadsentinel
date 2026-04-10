# base.py

"""
cadsentinel.etl.validators.base
--------------------------------
Shared validator contract, base class, and result builder.

Every deterministic validator:
  - Takes an evidence package (dict from retriever) and a rule config (dict)
  - Returns a ValidatorResult
  - Never raises — catches all errors and returns needs_review
  - Never calls an LLM or external API
  - Runs in < 100ms
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ── Result dataclass ─────────────────────────────────────────── #

@dataclass
class ValidatorResult:
    """
    Structured output from any deterministic validator.
    Maps directly to spec_execution_runs fields.
    """
    pass_fail:     str                      # pass | fail | warning | needs_review
    confidence:    float                    # 0.0 - 1.0
    severity:      str                      # low | medium | high | critical
    issue_summary: str                      # empty if pass
    issues:        list[dict]               # granular issues
    evidence_used: list[dict]               # evidence items consulted

    def to_dict(self) -> dict:
        return {
            "pass_fail":     self.pass_fail,
            "confidence":    self.confidence,
            "severity":      self.severity,
            "issue_summary": self.issue_summary,
            "issues":        self.issues,
            "evidence_used": self.evidence_used,
        }


# ── Issue builder ────────────────────────────────────────────── #

def make_issue(
    issue_type:    str,
    description:   str,
    severity:      str       = "medium",
    suggested_fix: str       = "",
    entity_ref:    Optional[dict] = None,
    confidence:    float     = 1.0,
) -> dict:
    """Build a single issue dict for ValidatorResult.issues."""
    return {
        "issue_type":    issue_type,
        "description":   description,
        "severity":      severity,
        "suggested_fix": suggested_fix,
        "entity_ref":    entity_ref,
        "confidence":    confidence,
    }


def make_evidence_ref(source: str, ref: str, value: Any = None) -> dict:
    """Build a single evidence reference dict."""
    return {
        "source": source,
        "ref":    ref,
        "value":  value,
    }


# ── Result constructors ──────────────────────────────────────── #

def pass_result(
    severity:      str       = "medium",
    evidence_used: list[dict] = None,
) -> ValidatorResult:
    return ValidatorResult(
        pass_fail     = "pass",
        confidence    = 1.0,
        severity      = severity,
        issue_summary = "",
        issues        = [],
        evidence_used = evidence_used or [],
    )


def fail_result(
    issue_summary: str,
    issues:        list[dict],
    severity:      str        = "medium",
    confidence:    float      = 1.0,
    evidence_used: list[dict] = None,
) -> ValidatorResult:
    return ValidatorResult(
        pass_fail     = "fail",
        confidence    = confidence,
        severity      = severity,
        issue_summary = issue_summary,
        issues        = issues,
        evidence_used = evidence_used or [],
    )


def warning_result(
    issue_summary: str,
    issues:        list[dict],
    severity:      str        = "low",
    confidence:    float      = 0.85,
    evidence_used: list[dict] = None,
) -> ValidatorResult:
    return ValidatorResult(
        pass_fail     = "warning",
        confidence    = confidence,
        severity      = severity,
        issue_summary = issue_summary,
        issues        = issues,
        evidence_used = evidence_used or [],
    )


def needs_review_result(
    reason:        str,
    severity:      str        = "medium",
    evidence_used: list[dict] = None,
) -> ValidatorResult:
    """Used when evidence is missing or ambiguous."""
    return ValidatorResult(
        pass_fail     = "needs_review",
        confidence    = 0.0,
        severity      = severity,
        issue_summary = reason,
        issues        = [make_issue("insufficient_evidence", reason, severity)],
        evidence_used = evidence_used or [],
    )


# ── Base validator class ─────────────────────────────────────── #

class BaseValidator:
    """
    Base class for all deterministic validators.

    Subclasses implement _validate(evidence, rule_config).
    The public validate() method wraps it with error handling.
    """

    name: str = "base"

    def validate(
        self,
        evidence:    dict,
        rule_config: dict,
    ) -> ValidatorResult:
        """
        Public entry point. Wraps _validate with a safety net.
        Never raises — returns needs_review on unexpected error.
        """
        try:
            return self._validate(evidence, rule_config)
        except Exception as exc:
            return needs_review_result(
                reason=f"Validator '{self.name}' raised an unexpected error: {exc}",
                severity=rule_config.get("severity_default", "medium"),
            )

    def _validate(
        self,
        evidence:    dict,
        rule_config: dict,
    ) -> ValidatorResult:
        raise NotImplementedError


# ── Evidence accessors ───────────────────────────────────────── #

def get_title_block(evidence: dict) -> Optional[dict]:
    """Extract the title block item from an evidence package."""
    for item in evidence.get("evidence", []):
        if item.get("source") == "drawing_title_block":
            return item
    return None


def get_layers(evidence: dict) -> list[dict]:
    """Extract all layer items from an evidence package."""
    return [
        item for item in evidence.get("evidence", [])
        if item.get("source") == "drawing_layers"
    ]


def get_dimensions(evidence: dict) -> list[dict]:
    """Extract all dimension items from an evidence package."""
    return [
        item for item in evidence.get("evidence", [])
        if item.get("source") == "drawing_dimensions"
    ]


def get_entities(evidence: dict) -> list[dict]:
    """Extract all entity items from an evidence package."""
    return [
        item for item in evidence.get("evidence", [])
        if item.get("source") in ("drawing_entities", "drawing_text_chunks")
    ]


def get_text_chunks(evidence: dict) -> list[dict]:
    """Extract text chunk items from an evidence package."""
    return [
        item for item in evidence.get("evidence", [])
        if item.get("source") == "drawing_text_chunks"
    ]

# ── Text collection utility ──────────────────────────────────── #

# AutoCAD control code substitutions
_AUTOCAD_SYMBOLS = {
    "%%p": "±",
    "%%d": "°",
    "%%c": "⌀",
    "%%u": "",
    "%%o": "",
}

import re as _re

# MTEXT formatting code patterns to strip
_MTEXT_FORMAT_PATTERN = _re.compile(
    r'\\A\d+;'           # \A1; alignment codes
    r'|\\[fF][^;]+;'     # \fFontName|...; or \Fromant.shx; font codes (case-insensitive)
    r'|\\C\d+;'          # \C6; color codes
    r'|\\H[\d.]+x?;'     # \H2.5; height codes
    r'|\\W[\d.]+;'       # \W1.0; width codes
    r'|\\Q[\d.]+;'       # \Q15; oblique codes
    r'|\\T[\d.]+;'       # \T1.0; tracking codes
    r'|\\S[^;^]*\^[^;]*;' # \S fraction codes
    r'|[{}]'             # curly braces
    r'|\\P'              # \P paragraph break
    r'|\\~'              # \~ non-breaking space
    r'|\\;'              # \; escape
)


def _strip_mtext(text: str) -> str:
    """Strip AutoCAD MTEXT formatting codes from a string."""
    cleaned = _MTEXT_FORMAT_PATTERN.sub(' ', text)
    # Substitute AutoCAD control codes
    for code, replacement in _AUTOCAD_SYMBOLS.items():
        cleaned = cleaned.replace(code, replacement)
    return cleaned


def collect_all_text(evidence: dict) -> str:
    parts: list[str] = []

    for item in evidence.get("evidence", []):
        source = item.get("source", "")

        if source in ("drawing_text_chunks", "drawing_entities"):
            text = (
                item.get("text") or
                item.get("chunk_text") or
                item.get("text_content") or
                ""
            )
            if text.strip():
                parts.append(_strip_mtext(text.strip()))

        elif source == "drawing_title_block":
            attrs = item.get("attributes") or {}
            for val in attrs.values():
                if val and str(val).strip():
                    parts.append(_strip_mtext(str(val).strip()))

    combined = " ".join(parts)
    return " ".join(combined.lower().split())