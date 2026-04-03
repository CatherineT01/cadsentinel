"""
cadsentinel.etl.spec_extractor
--------------------------------
LLM-based spec rule extraction and normalization.

Supports two providers via a common interface:
  - OpenAI  (GPT-4o by default)
  - Grok    (xAI, OpenAI-compatible API)

Configure via environment variables:
  CADSENTINEL_LLM_PROVIDER   = openai | grok      (default: openai)
  CADSENTINEL_LLM_MODEL      = model name          (default: gpt-4o)
  OPENAI_API_KEY             = your OpenAI key
  GROK_API_KEY               = your xAI/Grok key
  OPENAI_BASE_URL            = override for custom endpoints
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from .spec_models import (
    ExtractedSpec,
    NormalizedSpecRule,
    RuleType,
    ExecutionMode,
    Severity,
    infer_execution_mode,
    infer_evidence_types,
    build_retrieval_recipe,
    build_structured_rule,
)

log = logging.getLogger(__name__)

# ── Environment config ───────────────────────────────────────── #

LLM_PROVIDER = os.environ.get("CADSENTINEL_LLM_PROVIDER", "openai").lower()
LLM_MODEL    = os.environ.get("CADSENTINEL_LLM_MODEL", "gpt-4o")

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
GROK_API_KEY    = os.environ.get("GROK_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
GROK_BASE_URL   = os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1")

VALID_RULE_TYPES       = {r.value for r in RuleType}
VALID_EXECUTION_MODES  = {m.value for m in ExecutionMode}
VALID_SEVERITIES       = {s.value for s in Severity}
VALID_ENTITY_SCOPES    = {
    "dimensions", "notes", "title_block", "layers",
    "blocks", "text", "gdt", "bom",
}

# ── Extraction prompt ────────────────────────────────────────── #

EXTRACTION_SYSTEM_PROMPT = """You are an engineering QA system that extracts \
specification rules from technical documents.

Your job is to read a chunk of specification text and identify every individual, \
testable requirement within it. A requirement is a statement that a CAD drawing \
either passes or fails.

For each requirement found, return a JSON object with these fields:
- spec_code: section number or identifier if present (string or null)
- spec_title: short name for this requirement (string, max 80 chars)
- original_spec_text: the exact requirement text from the document (string)
- rule_type: one of: layer_naming, title_block, dimension_units, note_conformance, \
block_naming, text_style, revision_table, safety_note, cross_reference, general
- execution_mode: one of: deterministic, hybrid, llm_judge
- severity_default: one of: low, medium, high, critical
- entity_scope: list of affected drawing elements from: \
dimensions, notes, title_block, layers, blocks, text, gdt, bom
- extraction_confidence: float 0.0-1.0 (how confident you are this is a real requirement)

Return ONLY a JSON array of these objects. No preamble, no explanation, no markdown.
If no testable requirements are found in the text, return an empty array: []

Prefer deterministic execution_mode for concrete, measurable rules (layer names, \
field presence, unit type). Use hybrid for rules needing context. Use llm_judge \
only for inherently subjective requirements."""

EXTRACTION_USER_TEMPLATE = """Extract all testable specification requirements \
from the following text:

---
{chunk_text}
---

Return a JSON array only."""


# ── Abstract base class ──────────────────────────────────────── #

class BaseLLMClient(ABC):
    """Common interface for all LLM providers."""

    @abstractmethod
    def extract_specs(self, chunk_text: str) -> list[dict]:
        """
        Send chunk_text to the LLM and return a list of raw spec dicts.
        Each dict should match the fields in ExtractedSpec.
        Returns empty list on failure (never raises).
        """

    def _parse_llm_response(self, content: str) -> list[dict]:
        """
        Parse the LLM's JSON response into a list of dicts.
        Handles common formatting issues (markdown fences, trailing commas).
        """
        if not content or not content.strip():
            return []

        # Strip markdown code fences if present
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            ).strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            log.warning("LLM returned invalid JSON — attempting partial recovery")
            # Try to find a JSON array in the response
            start = text.find("[")
            end   = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    result = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    log.error("Could not parse LLM response as JSON")
                    return []
            else:
                return []

        if not isinstance(result, list):
            log.warning("LLM returned non-list JSON — wrapping in list")
            result = [result] if isinstance(result, dict) else []

        return result


# ── OpenAI client ────────────────────────────────────────────── #

class OpenAIClient(BaseLLMClient):
    """Calls the OpenAI Chat Completions API."""

    def __init__(self, model: str = LLM_MODEL, api_key: str = OPENAI_API_KEY):
        self.model   = model
        self.api_key = api_key

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set."
            )

        try:
            import openai
            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if OPENAI_BASE_URL:
                kwargs["base_url"] = OPENAI_BASE_URL
            self._client = openai.OpenAI(**kwargs)
        except ImportError as exc:
            raise RuntimeError(
                "openai package required: pip install openai"
            ) from exc

    def extract_specs(self, chunk_text: str) -> list[dict]:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": EXTRACTION_USER_TEMPLATE.format(
                        chunk_text=chunk_text
                    )},
                ],
                temperature=0.1,    # low temp for consistent structured output
                max_tokens=4096,
                response_format={"type": "json_object"} if "gpt-4" in self.model else None,
            )
            content = response.choices[0].message.content or ""
            return self._parse_llm_response(content)
        except Exception:
            log.exception("OpenAI extraction failed for chunk")
            return []


# ── Grok client ──────────────────────────────────────────────── #

class GrokClient(BaseLLMClient):
    """
    Calls the xAI Grok API.
    Grok uses an OpenAI-compatible API so we reuse the openai client
    with a different base_url and api_key.
    """

    def __init__(
        self,
        model:   str = "grok-2-latest",
        api_key: str = GROK_API_KEY,
    ):
        self.model   = model
        self.api_key = api_key

        if not self.api_key:
            raise ValueError(
                "GROK_API_KEY environment variable is not set."
            )

        try:
            import openai
            self._client = openai.OpenAI(
                api_key  = self.api_key,
                base_url = GROK_BASE_URL,
            )
        except ImportError as exc:
            raise RuntimeError(
                "openai package required for Grok client: pip install openai"
            ) from exc

    def extract_specs(self, chunk_text: str) -> list[dict]:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": EXTRACTION_USER_TEMPLATE.format(
                        chunk_text=chunk_text
                    )},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            content = response.choices[0].message.content or ""
            return self._parse_llm_response(content)
        except Exception:
            log.exception("Grok extraction failed for chunk")
            return []


# ── Provider factory ─────────────────────────────────────────── #

def get_llm_client(provider: str = LLM_PROVIDER) -> BaseLLMClient:
    """
    Return the configured LLM client.

    Args:
        provider: "openai" or "grok". Defaults to CADSENTINEL_LLM_PROVIDER env var.
    """
    provider = provider.lower()
    if provider == "openai":
        return OpenAIClient()
    elif provider == "grok":
        return GrokClient()
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Set CADSENTINEL_LLM_PROVIDER to 'openai' or 'grok'."
        )


# ── Validation + normalization ───────────────────────────────── #

def validate_raw_spec(raw: dict) -> bool:
    """Return True if the raw dict has the minimum required fields."""
    return bool(raw.get("original_spec_text", "").strip())


def normalize_raw_spec(raw: dict, chunk_index: int) -> ExtractedSpec:
    """
    Convert a raw LLM-output dict into a validated ExtractedSpec.
    Fills defaults for missing or invalid fields.
    """
    rule_type = raw.get("rule_type", RuleType.GENERAL.value)
    if rule_type not in VALID_RULE_TYPES:
        log.debug("Unknown rule_type '%s' — defaulting to 'general'", rule_type)
        rule_type = RuleType.GENERAL.value

    execution_mode = raw.get("execution_mode") or infer_execution_mode(rule_type)
    if execution_mode not in VALID_EXECUTION_MODES:
        execution_mode = infer_execution_mode(rule_type)

    severity = raw.get("severity_default", Severity.MEDIUM.value)
    if severity not in VALID_SEVERITIES:
        severity = Severity.MEDIUM.value

    raw_scope = raw.get("entity_scope", [])
    if isinstance(raw_scope, str):
        raw_scope = [raw_scope]
    entity_scope = [s for s in raw_scope if s in VALID_ENTITY_SCOPES]
    if not entity_scope:
        entity_scope = ["notes"]

    confidence = float(raw.get("extraction_confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    return ExtractedSpec(
        original_spec_text    = raw["original_spec_text"].strip(),
        spec_code             = raw.get("spec_code"),
        spec_title            = raw.get("spec_title"),
        rule_type             = rule_type,
        execution_mode        = execution_mode,
        severity_default      = severity,
        entity_scope          = entity_scope,
        extraction_confidence = confidence,
        source_chunk_index    = chunk_index,
    )


def build_normalized_rule(
    spec: ExtractedSpec,
    spec_document_id: int,
) -> NormalizedSpecRule:
    """
    Convert a validated ExtractedSpec into a NormalizedSpecRule
    ready for database insertion.
    """
    retrieval_recipe  = build_retrieval_recipe(spec.entity_scope, spec.rule_type)
    structured_rule   = build_structured_rule(spec)
    expected_evidence = infer_evidence_types(spec.entity_scope)

    # Normalized rule text — cleaned version of the original
    normalized_text = spec.original_spec_text.strip()
    if spec.spec_title and spec.spec_title not in normalized_text:
        normalized_text = f"{spec.spec_title}: {normalized_text}"

    return NormalizedSpecRule(
        spec_document_id        = spec_document_id,
        original_spec_text      = spec.original_spec_text,
        normalized_rule_text    = normalized_text,
        spec_code               = spec.spec_code,
        spec_title              = spec.spec_title,
        rule_type               = spec.rule_type,
        execution_mode          = spec.execution_mode,
        severity_default        = spec.severity_default,
        entity_scope            = spec.entity_scope,
        expected_evidence_types = expected_evidence,
        checker_prompt          = None,
        retrieval_recipe        = retrieval_recipe,
        structured_rule         = structured_rule,
        extraction_confidence   = spec.extraction_confidence,
        normalization_confidence= 0.80,
        source_chunk_index      = spec.source_chunk_index,
    )


# ── Main extraction pipeline ─────────────────────────────────── #

def extract_specs_from_chunks(
    chunks: list,
    spec_document_id: int,
    client: BaseLLMClient,
) -> list[NormalizedSpecRule]:
    """
    Run extraction over all chunks and return normalized rules.
    Skips chunks that produce no valid specs.
    Continues on per-chunk failures.
    """
    all_rules: list[NormalizedSpecRule] = []

    for chunk in chunks:
        log.info(
            "Extracting specs from chunk %d (%d chars)",
            chunk.chunk_index, len(chunk.raw_text)
        )

        raw_specs = client.extract_specs(chunk.raw_text)

        if not raw_specs:
            log.debug("No specs found in chunk %d", chunk.chunk_index)
            continue

        for raw in raw_specs:
            if not validate_raw_spec(raw):
                log.debug("Skipping invalid spec from chunk %d", chunk.chunk_index)
                continue

            try:
                extracted   = normalize_raw_spec(raw, chunk.chunk_index)
                normalized  = build_normalized_rule(extracted, spec_document_id)
                all_rules.append(normalized)
            except Exception:
                log.exception(
                    "Failed to normalize spec from chunk %d", chunk.chunk_index
                )

    log.info(
        "Extracted %d normalized rules from %d chunks",
        len(all_rules), len(chunks)
    )
    return all_rules