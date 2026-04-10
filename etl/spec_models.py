# spec_models.py

"""
cadsentinel.etl.spec_models
----------------------------
Data classes for the spec ingestion pipeline.
These are the internal representations used between parsing,
extraction, normalization, and database insertion.
No database logic here — pure data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ExecutionMode(str, Enum):
    DETERMINISTIC = "deterministic"
    HYBRID        = "hybrid"
    LLM_JUDGE     = "llm_judge"


class Severity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class RuleType(str, Enum):
    LAYER_NAMING      = "layer_naming"
    TITLE_BLOCK       = "title_block"
    DIMENSION_UNITS   = "dimension_units"
    NOTE_CONFORMANCE  = "note_conformance"
    BLOCK_NAMING      = "block_naming"
    TEXT_STYLE        = "text_style"
    REVISION_TABLE    = "revision_table"
    SAFETY_NOTE       = "safety_note"
    CROSS_REFERENCE   = "cross_reference"
    GENERAL           = "general"
    MODEL_CODE      = "model_code"
    STANDARD_NOTES  = "standard_notes"
    CYLINDER_SPEC   = "cylinder_spec"
    JIT_BORE        = "jit_bore"
    JIT_MOUNT       = "jit_mount"


@dataclass
class SpecChunk:
    """
    A chunk of text extracted from a specification document.
    Stored in spec_document_sections.
    """
    chunk_index:  int
    raw_text:     str
    char_start:   int
    char_end:     int
    source_page:  Optional[int] = None   # PDF page number if available
    section_hint: Optional[str] = None   # heading text if detected


@dataclass
class ExtractedSpec:
    """
    A single specification rule as returned by the LLM extractor.
    Not yet normalized or stored — intermediate representation.
    """
    original_spec_text:    str
    spec_code:             Optional[str]  = None
    spec_title:            Optional[str]  = None
    rule_type:             str            = RuleType.GENERAL.value
    execution_mode:        str            = ExecutionMode.HYBRID.value
    severity_default:      str            = Severity.MEDIUM.value
    entity_scope:          list[str]      = field(default_factory=list)
    extraction_confidence: float          = 0.5
    source_chunk_index:    Optional[int]  = None


@dataclass
class NormalizedSpecRule:
    """
    A fully normalized spec rule ready to be written to spec_rules.
    """
    spec_document_id:       int
    original_spec_text:     str
    normalized_rule_text:   str
    spec_code:              Optional[str]
    spec_title:             Optional[str]
    rule_type:              str
    execution_mode:         str
    severity_default:       str
    entity_scope:           list[str]
    expected_evidence_types: list[str]
    checker_prompt:         Optional[str]
    retrieval_recipe:       dict
    structured_rule:        dict
    extraction_confidence:  float
    normalization_confidence: float
    source_chunk_index:     Optional[int]


@dataclass
class SpecDocument:
    """
    Metadata for an uploaded specification document.
    Stored in spec_documents.
    """
    filename:          str
    source_type:       str            # pdf, docx, txt
    raw_text:          str
    title:             Optional[str]  = None
    organization_name: Optional[str]  = None
    discipline:        Optional[str]  = None
    storage_path:      Optional[str]  = None


# ── Helpers ───────────────────────────────────────────────────── #

# Maps entity scope keywords to expected evidence types
SCOPE_TO_EVIDENCE: dict[str, list[str]] = {
    "dimensions":  ["dimension"],
    "notes":       ["note", "text"],
    "title_block": ["title_block"],
    "layers":      ["layer"],
    "blocks":      ["insert", "block"],
    "text":        ["text", "note"],
}

# Maps rule types to their default execution modes
RULE_TYPE_EXECUTION_DEFAULTS: dict[str, str] = {
    RuleType.LAYER_NAMING.value:     ExecutionMode.DETERMINISTIC.value,
    RuleType.TITLE_BLOCK.value:      ExecutionMode.DETERMINISTIC.value,
    RuleType.DIMENSION_UNITS.value:  ExecutionMode.DETERMINISTIC.value,
    RuleType.BLOCK_NAMING.value:     ExecutionMode.DETERMINISTIC.value,
    RuleType.TEXT_STYLE.value:       ExecutionMode.DETERMINISTIC.value,
    RuleType.REVISION_TABLE.value:   ExecutionMode.DETERMINISTIC.value,
    RuleType.NOTE_CONFORMANCE.value: ExecutionMode.HYBRID.value,
    RuleType.CROSS_REFERENCE.value:  ExecutionMode.HYBRID.value,
    RuleType.SAFETY_NOTE.value:      ExecutionMode.LLM_JUDGE.value,
    RuleType.GENERAL.value:          ExecutionMode.HYBRID.value,
    RuleType.MODEL_CODE.value:     ExecutionMode.DETERMINISTIC.value,
    RuleType.STANDARD_NOTES.value: ExecutionMode.DETERMINISTIC.value,
    RuleType.CYLINDER_SPEC.value:  ExecutionMode.DETERMINISTIC.value,
    RuleType.JIT_BORE.value:       ExecutionMode.DETERMINISTIC.value,
    RuleType.JIT_MOUNT.value:     ExecutionMode.DETERMINISTIC.value,
}


def infer_execution_mode(rule_type: str) -> str:
    """Return the default execution mode for a given rule type."""
    return RULE_TYPE_EXECUTION_DEFAULTS.get(
        rule_type, ExecutionMode.HYBRID.value
    )


def infer_evidence_types(entity_scope: list[str]) -> list[str]:
    """Derive expected evidence types from entity scope."""
    evidence: list[str] = []
    for scope in entity_scope:
        evidence.extend(SCOPE_TO_EVIDENCE.get(scope, []))
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for e in evidence:
        if e not in seen:
            seen.add(e)
            result.append(e)
    return result or ["text"]


def build_retrieval_recipe(
    entity_scope: list[str],
    rule_type: str,
    top_k: int = 8,
) -> dict:
    """
    Build a default retrieval_recipe from entity scope and rule type.
    The recipe can be refined later during human review.
    """
    source_map = {
        "dimensions":  "dimension",
        "notes":       "note",
        "title_block": "title_block",
        "layers":      "layer",
        "blocks":      "entity",
        "text":        "note",
    }

    source_types = list({
        source_map.get(s, "note")
        for s in entity_scope
    }) or ["note"]

    # Layer-naming and title-block rules always include those sources
    if rule_type == RuleType.LAYER_NAMING.value and "layer" not in source_types:
        source_types.append("layer")
    if rule_type == RuleType.TITLE_BLOCK.value and "title_block" not in source_types:
        source_types.append("title_block")

    return {
        "source_types":    source_types,
        "top_k":           top_k,
        "keyword_filters": [],
        "entity_filters":  {},
    }


def build_structured_rule(spec: ExtractedSpec) -> dict:
    """
    Build the structured_rule JSON object from an ExtractedSpec.
    This is the machine-readable rule definition stored in spec_rules.
    """
    requires_llm = spec.execution_mode == ExecutionMode.LLM_JUDGE.value

    return {
        "rule_type":            spec.rule_type,
        "description":          spec.original_spec_text[:500],
        "entity_scope":         spec.entity_scope,
        "pass_condition":       {},
        "fail_condition":       {},
        "expected_evidence":    infer_evidence_types(spec.entity_scope),
        "severity_default":     spec.severity_default,
        "requires_llm_reasoning": requires_llm,
    }