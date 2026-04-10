# llm_checker.py

"""
cadsentinel.etl.execution.llm_checker
---------------------------------------
LLM-based spec checking for hybrid and llm_judge execution modes.

Both modes use the same structured JSON output contract.
The difference is what gets sent:
  - hybrid:    evidence package retrieved by the retriever + rule text
  - llm_judge: same, but the checker_prompt is more specific and interpretive

Provider selection mirrors spec_extractor.py — same env vars, same pattern.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

log = logging.getLogger(__name__)

LLM_PROVIDER    = os.environ.get("CADSENTINEL_LLM_PROVIDER", "openai").lower()
LLM_MODEL       = os.environ.get("CADSENTINEL_LLM_MODEL", "gpt-4o")
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
GROK_API_KEY    = os.environ.get("GROK_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
GROK_BASE_URL   = os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1")

# ── System prompt ────────────────────────────────────────────── #

CHECKER_SYSTEM_PROMPT = """You are an engineering QA system evaluating whether \
an AutoCAD drawing complies with a specific engineering specification.

You will be given:
1. The specification rule to check
2. Relevant drawing evidence (text notes, dimensions, title block data, layer names)

Your job is to determine if the drawing PASSES or FAILS this specific rule.

Return ONLY a JSON object with this exact structure:
{
  "pass_fail": "pass" | "fail" | "warning" | "needs_review",
  "confidence": 0.0-1.0,
  "severity": "low" | "medium" | "high" | "critical",
  "issue_summary": "brief description of the problem (empty string if pass)",
  "issues": [
    {
      "issue_type": "string",
      "description": "detailed description",
      "evidence": ["list of evidence items that support this finding"],
      "suggested_fix": "what the drafter should do to fix this"
    }
  ],
  "evidence_used": [
    {"type": "string", "ref": "entity handle or identifier", "value": "content"}
  ]
}

Rules:
- Only use the evidence provided. Do not invent findings not supported by evidence.
- If evidence is insufficient to make a determination, return "needs_review".
- confidence should reflect how certain you are, not how severe the issue is.
- issues array must be empty if pass_fail is "pass".
- Be specific — cite the exact evidence that led to your conclusion.
- Return ONLY the JSON object. No preamble, no explanation."""

CHECKER_USER_TEMPLATE = """SPECIFICATION RULE:
{rule_text}

RULE TYPE: {rule_type}
SEVERITY: {severity}

DRAWING EVIDENCE:
{evidence_text}

Evaluate whether this drawing complies with the specification rule above.
Return only the JSON result object."""


# ── Evidence formatter ───────────────────────────────────────── #

def format_evidence_for_prompt(evidence_package: dict) -> str:
    """
    Convert an evidence package into readable text for the LLM prompt.
    Groups evidence by source type for clarity.
    """
    items = evidence_package.get("evidence", [])
    if not items:
        return "No evidence available."

    sections: dict[str, list[str]] = {}

    for item in items:
        source = item.get("source", "unknown")

        if source == "drawing_title_block":
            attrs = item.get("attributes") or {}
            if attrs:
                lines = [f"  {k}: {v}" for k, v in attrs.items()]
                sections.setdefault("Title Block", []).extend(lines)

        elif source == "drawing_dimensions":
            val  = item.get("measured_value")
            text = item.get("user_text") or ""
            handle = item.get("entity_handle", "")
            layer  = item.get("layer", "")
            desc   = f"  [{handle}] {item.get('dim_type','')} = {val}"
            if text:
                desc += f" (text: '{text}')"
            if layer:
                desc += f" layer: {layer}"
            sections.setdefault("Dimensions", []).append(desc)

        elif source == "drawing_layers":
            name = item.get("layer_name", "")
            sections.setdefault("Layers", []).append(f"  {name}")

        elif source in ("drawing_text_chunks", "drawing_entities"):
            text = item.get("text") or item.get("chunk_text") or ""
            if text.strip():
                handle = item.get("entity_handle", "")
                score  = item.get("similarity_score")
                desc   = f"  [{handle}] {text[:200]}"
                if score is not None:
                    desc += f" (relevance: {score:.2f})"
                sections.setdefault("Notes & Text", []).append(desc)

    if not sections:
        return "Evidence retrieved but no readable content found."

    lines: list[str] = []
    for section_name, content in sections.items():
        lines.append(f"{section_name}:")
        lines.extend(content)
        lines.append("")

    return "\n".join(lines).strip()


# ── LLM client (shared pattern with spec_extractor) ──────────── #

def _get_openai_client():
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("pip install openai") from exc

    kwargs: dict[str, Any] = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return openai.OpenAI(**kwargs)


def _get_grok_client():
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("pip install openai") from exc

    return openai.OpenAI(api_key=GROK_API_KEY, base_url=GROK_BASE_URL)


def _get_client(provider: str):
    if provider == "openai":
        return _get_openai_client()
    elif provider == "grok":
        return _get_grok_client()
    raise ValueError(f"Unknown provider: {provider}")


def _get_model(provider: str) -> str:
    if provider == "grok":
        return os.environ.get("CADSENTINEL_LLM_MODEL", "grok-2-latest")
    return LLM_MODEL


# ── Core checker function ────────────────────────────────────── #

def llm_check(
    evidence_package:  dict,
    rule_text:         str,
    rule_type:         str,
    severity:          str,
    checker_prompt:    Optional[str] = None,
    provider:          str           = LLM_PROVIDER,
    temperature:       float         = 0.1,
) -> dict:
    """
    Run an LLM check for one spec rule against one evidence package.

    Args:
        evidence_package: Output from EvidenceRetriever.retrieve()
        rule_text:        normalized_rule_text from spec_rules
        rule_type:        e.g. "note_conformance", "safety_note"
        severity:         severity_default from spec_rules
        checker_prompt:   optional override — used as rule_text if provided
        provider:         "openai" or "grok"
        temperature:      LLM temperature (keep low for consistency)

    Returns:
        dict with pass_fail, confidence, severity, issue_summary,
        issues, evidence_used, model_name, token_input, token_output,
        latency_ms
    """
    effective_rule_text = checker_prompt or rule_text
    evidence_text       = format_evidence_for_prompt(evidence_package)
    model               = _get_model(provider)

    user_message = CHECKER_USER_TEMPLATE.format(
        rule_text     = effective_rule_text,
        rule_type     = rule_type,
        severity      = severity,
        evidence_text = evidence_text,
    )

    start_ms = int(time.time() * 1000)

    try:
        client   = _get_client(provider)
        response = client.chat.completions.create(
            model       = model,
            messages    = [
                {"role": "system", "content": CHECKER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature = temperature,
            max_tokens  = 2048,
        )

        latency_ms   = int(time.time() * 1000) - start_ms
        content      = response.choices[0].message.content or ""
        token_input  = response.usage.prompt_tokens     if response.usage else 0
        token_output = response.usage.completion_tokens if response.usage else 0

        result = _parse_llm_result(content, severity)
        result["model_name"]    = model
        result["token_input"]   = token_input
        result["token_output"]  = token_output
        result["latency_ms"]    = latency_ms
        result["provider"]      = provider

        return result

    except Exception as exc:
        latency_ms = int(time.time() * 1000) - start_ms
        log.exception("LLM check failed for rule_type='%s'", rule_type)
        return _error_result(str(exc), severity, model, latency_ms)


# ── Response parsing ─────────────────────────────────────────── #

def _parse_llm_result(content: str, severity: str) -> dict:
    """
    Parse the LLM JSON response into a validated result dict.
    Falls back to needs_review on any parse failure.
    """
    text = content.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        ).strip()

    # Find JSON object
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        log.warning("LLM response contained no JSON object")
        return _error_result("No JSON in response", severity, "", 0)

    try:
        raw = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        log.warning("LLM JSON parse failed: %s", exc)
        return _error_result(f"JSON parse error: {exc}", severity, "", 0)

    # Validate and normalise fields
    pass_fail = raw.get("pass_fail", "needs_review")
    if pass_fail not in ("pass", "fail", "warning", "needs_review"):
        pass_fail = "needs_review"

    confidence = float(raw.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    result_severity = raw.get("severity", severity)
    if result_severity not in ("low", "medium", "high", "critical"):
        result_severity = severity

    issues = raw.get("issues", [])
    if not isinstance(issues, list):
        issues = []

    # Sanitise each issue
    clean_issues = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        clean_issues.append({
            "issue_type":    str(issue.get("issue_type", "unknown")),
            "description":   str(issue.get("description", "")),
            "evidence":      issue.get("evidence", []),
            "suggested_fix": str(issue.get("suggested_fix", "")),
        })

    evidence_used = raw.get("evidence_used", [])
    if not isinstance(evidence_used, list):
        evidence_used = []

    return {
        "pass_fail":     pass_fail,
        "confidence":    confidence,
        "severity":      result_severity,
        "issue_summary": str(raw.get("issue_summary", "")),
        "issues":        clean_issues,
        "evidence_used": evidence_used,
    }


def _error_result(
    reason:     str,
    severity:   str,
    model_name: str,
    latency_ms: int,
) -> dict:
    return {
        "pass_fail":     "needs_review",
        "confidence":    0.0,
        "severity":      severity,
        "issue_summary": f"LLM check failed: {reason}",
        "issues":        [{
            "issue_type":    "llm_error",
            "description":   reason,
            "evidence":      [],
            "suggested_fix": "Retry or escalate to human review.",
        }],
        "evidence_used": [],
        "model_name":    model_name,
        "token_input":   0,
        "token_output":  0,
        "latency_ms":    latency_ms,
        "provider":      "",
    }