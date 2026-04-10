# test_spec_ingestor.py

"""
tests/test_spec_ingestor.py
----------------------------
Unit tests for the spec ingestion pipeline.
Tests parser, extractor, models, and ingestor logic.
No real files, database, or LLM API required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cadsentinel.etl.spec_models import (
    ExtractedSpec,
    RuleType,
    ExecutionMode,
    Severity,
    infer_execution_mode,
    infer_evidence_types,
    build_retrieval_recipe,
    build_structured_rule,
    RULE_TYPE_EXECUTION_DEFAULTS,
)
from cadsentinel.etl.spec_parser import (
    _chunk_text,
    _detect_heading,
    _infer_title,
    ParseError,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from cadsentinel.etl.spec_extractor import (
    validate_raw_spec,
    normalize_raw_spec,
    build_normalized_rule,
    BaseLLMClient,
    VALID_RULE_TYPES,
    VALID_EXECUTION_MODES,
    VALID_SEVERITIES,
)
from cadsentinel.etl.spec_ingestor import SpecIngestor, SpecIngestionError


# ── spec_models tests ─────────────────────────────────────────── #

class TestSpecModels:
    def test_infer_execution_mode_deterministic_for_layer_naming(self):
        assert infer_execution_mode("layer_naming") == "deterministic"

    def test_infer_execution_mode_deterministic_for_title_block(self):
        assert infer_execution_mode("title_block") == "deterministic"

    def test_infer_execution_mode_hybrid_for_note_conformance(self):
        assert infer_execution_mode("note_conformance") == "hybrid"

    def test_infer_execution_mode_llm_judge_for_safety_note(self):
        assert infer_execution_mode("safety_note") == "llm_judge"

    def test_infer_execution_mode_defaults_to_hybrid(self):
        assert infer_execution_mode("unknown_type") == "hybrid"

    def test_infer_evidence_types_from_dimensions(self):
        result = infer_evidence_types(["dimensions"])
        assert "dimension" in result

    def test_infer_evidence_types_from_notes(self):
        result = infer_evidence_types(["notes"])
        assert "note" in result or "text" in result

    def test_infer_evidence_types_deduplicates(self):
        result = infer_evidence_types(["notes", "text"])
        assert len(result) == len(set(result))

    def test_infer_evidence_types_returns_text_for_empty(self):
        result = infer_evidence_types([])
        assert result == ["text"]

    def test_build_retrieval_recipe_has_required_keys(self):
        recipe = build_retrieval_recipe(["dimensions"], "dimension_units")
        assert "source_types"    in recipe
        assert "top_k"           in recipe
        assert "keyword_filters" in recipe
        assert "entity_filters"  in recipe

    def test_build_retrieval_recipe_layer_naming_includes_layer(self):
        recipe = build_retrieval_recipe([], "layer_naming")
        assert "layer" in recipe["source_types"]

    def test_build_retrieval_recipe_title_block_includes_title_block(self):
        recipe = build_retrieval_recipe([], "title_block")
        assert "title_block" in recipe["source_types"]

    def test_build_structured_rule_has_required_keys(self):
        spec = ExtractedSpec(
            original_spec_text="All layers must follow naming convention.",
            rule_type="layer_naming",
            execution_mode="deterministic",
            entity_scope=["layers"],
        )
        rule = build_structured_rule(spec)
        assert "rule_type"              in rule
        assert "description"            in rule
        assert "entity_scope"           in rule
        assert "severity_default"       in rule
        assert "requires_llm_reasoning" in rule

    def test_build_structured_rule_llm_false_for_deterministic(self):
        spec = ExtractedSpec(
            original_spec_text="Layer names must start with DIM.",
            rule_type="layer_naming",
            execution_mode="deterministic",
            entity_scope=["layers"],
        )
        rule = build_structured_rule(spec)
        assert rule["requires_llm_reasoning"] is False

    def test_build_structured_rule_llm_true_for_llm_judge(self):
        spec = ExtractedSpec(
            original_spec_text="Safety note must be clear.",
            rule_type="safety_note",
            execution_mode="llm_judge",
            entity_scope=["notes"],
        )
        rule = build_structured_rule(spec)
        assert rule["requires_llm_reasoning"] is True


# ── spec_parser tests ─────────────────────────────────────────── #

class TestSpecParser:
    def test_chunk_text_basic(self):
        text = "A" * 5000
        chunks = _chunk_text(text, {})
        assert len(chunks) >= 2

    def test_chunk_size_respected(self):
        text = "B" * 5000
        chunks = _chunk_text(text, {})
        for chunk in chunks:
            assert len(chunk.raw_text) <= CHUNK_SIZE + 100

    def test_chunk_index_sequential(self):
        text = "C" * 5000
        chunks = _chunk_text(text, {})
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_short_text_produces_one_chunk(self):
        text = "This is a short specification document with enough content to pass the minimum length check."
        chunks = _chunk_text(text, {})
        assert len(chunks) == 1

    def test_empty_text_produces_no_chunks(self):
        chunks = _chunk_text("", {})
        assert len(chunks) == 0

    def test_very_short_text_below_min_len_discarded(self):
        chunks = _chunk_text("Hi", {})
        assert len(chunks) == 0

    def test_detect_heading_short_line_no_period(self):
        text = "3.2 Layer Naming Requirements\nAll layers must..."
        heading = _detect_heading(text)
        assert heading == "3.2 Layer Naming Requirements"

    def test_detect_heading_returns_none_for_long_line(self):
        text = ("A" * 150) + "\nSome content"
        heading = _detect_heading(text)
        assert heading is None

    def test_detect_heading_returns_none_for_sentence(self):
        text = "All dimensions must be shown in inches.\nMore text."
        heading = _detect_heading(text)
        assert heading is None

    def test_infer_title_from_first_line(self):
        text = "ASTM Drawing Standards\nSection 1: General Requirements"
        title = _infer_title(text, "astm_standards.pdf")
        assert title == "ASTM Drawing Standards"

    def test_infer_title_falls_back_to_filename(self):
        text = "This is a long sentence that ends with a period.\nMore content."
        title = _infer_title(text, "my_spec_document.pdf")
        assert "my" in title.lower() or "spec" in title.lower()

    def test_parse_txt_file(self, tmp_path):
        from cadsentinel.etl.spec_parser import parse_document
        spec_file = tmp_path / "test_spec.txt"
        spec_file.write_text(
            "3.1 Layer Naming\nAll layers must be named according to company standard.\n"
            "3.2 Title Block\nThe title block must contain a drawing number.",
            encoding="utf-8"
        )
        doc, chunks = parse_document(spec_file)
        assert doc.source_type == "txt"
        assert len(doc.raw_text) > 0
        assert len(chunks) >= 1

    def test_parse_raises_for_unsupported_format(self, tmp_path):
        from cadsentinel.etl.spec_parser import parse_document
        bad_file = tmp_path / "drawing.dwg"
        bad_file.write_bytes(b"\x00" * 100)
        with pytest.raises(ParseError, match="Unsupported file format"):
            parse_document(bad_file)

    def test_parse_raises_for_missing_file(self, tmp_path):
        from cadsentinel.etl.spec_parser import parse_document
        with pytest.raises(ParseError, match="File not found"):
            parse_document(tmp_path / "nonexistent.txt")


# ── spec_extractor tests ──────────────────────────────────────── #

class TestSpecExtractor:
    def test_validate_raw_spec_passes_with_text(self):
        raw = {"original_spec_text": "All dimensions must be in inches."}
        assert validate_raw_spec(raw) is True

    def test_validate_raw_spec_fails_with_empty_text(self):
        assert validate_raw_spec({"original_spec_text": ""}) is False

    def test_validate_raw_spec_fails_with_missing_text(self):
        assert validate_raw_spec({}) is False

    def test_normalize_raw_spec_defaults_for_unknown_rule_type(self):
        raw = {
            "original_spec_text": "Some requirement.",
            "rule_type": "completely_unknown",
        }
        spec = normalize_raw_spec(raw, chunk_index=0)
        assert spec.rule_type == "general"

    def test_normalize_raw_spec_defaults_for_unknown_severity(self):
        raw = {
            "original_spec_text": "Some requirement.",
            "severity_default": "ultra_critical",
        }
        spec = normalize_raw_spec(raw, chunk_index=0)
        assert spec.severity_default == "medium"

    def test_normalize_raw_spec_defaults_for_invalid_execution_mode(self):
        raw = {
            "original_spec_text": "Some requirement.",
            "execution_mode": "magic",
        }
        spec = normalize_raw_spec(raw, chunk_index=0)
        assert spec.execution_mode in VALID_EXECUTION_MODES

    def test_normalize_raw_spec_clamps_confidence(self):
        raw = {
            "original_spec_text": "Some requirement.",
            "extraction_confidence": 1.5,
        }
        spec = normalize_raw_spec(raw, chunk_index=0)
        assert spec.extraction_confidence <= 1.0

    def test_normalize_raw_spec_filters_invalid_entity_scope(self):
        raw = {
            "original_spec_text": "Some requirement.",
            "entity_scope": ["dimensions", "invalid_scope", "notes"],
        }
        spec = normalize_raw_spec(raw, chunk_index=0)
        assert "invalid_scope" not in spec.entity_scope

    def test_normalize_raw_spec_defaults_entity_scope_to_notes(self):
        raw = {
            "original_spec_text": "Some requirement.",
            "entity_scope": [],
        }
        spec = normalize_raw_spec(raw, chunk_index=0)
        assert spec.entity_scope == ["notes"]

    def test_build_normalized_rule_has_all_fields(self):
        spec = ExtractedSpec(
            original_spec_text="All layers must be named DIM-*.",
            rule_type="layer_naming",
            execution_mode="deterministic",
            entity_scope=["layers"],
            extraction_confidence=0.92,
            spec_code="3.1",
            spec_title="Layer Naming",
        )
        rule = build_normalized_rule(spec, spec_document_id=1)
        assert rule.spec_document_id   == 1
        assert rule.rule_type          == "layer_naming"
        assert rule.execution_mode     == "deterministic"
        assert rule.retrieval_recipe   is not None
        assert rule.structured_rule    is not None
        assert rule.approved_at        if hasattr(rule, 'approved_at') else True

    def test_parse_llm_response_handles_json_array(self):
        class TestClient(BaseLLMClient):
            def extract_specs(self, chunk_text): return []
        client = TestClient()
        result = client._parse_llm_response(
            '[{"original_spec_text": "All dims in inches.", "rule_type": "dimension_units"}]'
        )
        assert len(result) == 1
        assert result[0]["rule_type"] == "dimension_units"

    def test_parse_llm_response_handles_markdown_fences(self):
        class TestClient(BaseLLMClient):
            def extract_specs(self, chunk_text): return []
        client = TestClient()
        raw = '```json\n[{"original_spec_text": "Rule text."}]\n```'
        result = client._parse_llm_response(raw)
        assert len(result) == 1

    def test_parse_llm_response_returns_empty_for_invalid_json(self):
        class TestClient(BaseLLMClient):
            def extract_specs(self, chunk_text): return []
        client = TestClient()
        result = client._parse_llm_response("not json at all")
        assert result == []

    def test_parse_llm_response_returns_empty_for_empty_string(self):
        class TestClient(BaseLLMClient):
            def extract_specs(self, chunk_text): return []
        client = TestClient()
        assert client._parse_llm_response("") == []


# ── spec_ingestor tests ───────────────────────────────────────── #

def make_mock_conn_cur():
    cur = MagicMock()
    cur.fetchone.return_value = {"id": 1}
    conn = MagicMock()
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__  = MagicMock(return_value=False)
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
    conn.cursor.return_value.__exit__  = MagicMock(return_value=False)
    return conn, cur


class TestSpecIngestor:
    def test_raises_if_file_missing(self, tmp_path):
        ingestor = SpecIngestor(skip_extraction=True)
        with pytest.raises(SpecIngestionError, match="File not found"):
            ingestor.ingest(tmp_path / "missing.txt")

    def test_raises_if_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "drawing.dwg"
        bad_file.write_bytes(b"\x00" * 100)
        ingestor = SpecIngestor(skip_extraction=True)
        with pytest.raises(SpecIngestionError, match="Parse failed"):
            ingestor.ingest(bad_file)

    def test_ingest_txt_skip_extraction(self, tmp_path):
        spec_file = tmp_path / "spec.txt"
        spec_file.write_text(
            "3.1 Layer Naming\nAll layers must be named according to convention.\n"
            "3.2 Title Block\nTitle block must contain drawing number and revision.",
            encoding="utf-8"
        )
        conn, cur = make_mock_conn_cur()
        # First fetchone: no duplicate found; second: returns inserted id
        cur.fetchone.side_effect = [None, {"id": 42}]

        with patch("cadsentinel.etl.spec_ingestor.get_connection") as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__  = MagicMock(return_value=False)

            ingestor = SpecIngestor(skip_extraction=True)
            result   = ingestor.ingest(spec_file)

        assert result["skipped"]          is False
        assert result["rules_extracted"]  == 0   # extraction skipped
        assert result["chunks_stored"]    >= 1

    def test_skips_duplicate_file(self, tmp_path):
        spec_file = tmp_path / "spec.txt"
        spec_file.write_text("Some spec content here for testing.", encoding="utf-8")
        conn, cur = make_mock_conn_cur()
        # fetchone returns existing id → duplicate detected
        cur.fetchone.return_value = {"id": 99}

        with patch("cadsentinel.etl.spec_ingestor.get_connection") as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__  = MagicMock(return_value=False)

            ingestor = SpecIngestor(skip_extraction=True)
            result   = ingestor.ingest(spec_file)

        assert result["skipped"]          is True
        assert result["spec_document_id"] == 99

    def test_result_has_required_keys(self, tmp_path):
        spec_file = tmp_path / "spec.txt"
        spec_file.write_text("Requirement: all layers must be named correctly.", encoding="utf-8")
        conn, cur = make_mock_conn_cur()
        cur.fetchone.side_effect = [None, {"id": 5}]

        with patch("cadsentinel.etl.spec_ingestor.get_connection") as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__  = MagicMock(return_value=False)

            ingestor = SpecIngestor(skip_extraction=True)
            result   = ingestor.ingest(spec_file)

        assert "spec_document_id" in result
        assert "chunks_stored"    in result
        assert "rules_extracted"  in result
        assert "rules_stored"     in result
        assert "skipped"          in result

    def test_ingest_with_mock_llm(self, tmp_path):
        """Verify that extracted rules get inserted when LLM returns specs."""
        spec_file = tmp_path / "spec.txt"
        spec_file.write_text(
            "All layer names must start with DIM for dimension layers.",
            encoding="utf-8"
        )

        # Mock LLM client
        mock_client = MagicMock()
        mock_client.extract_specs.return_value = [
            {
                "original_spec_text": "All layer names must start with DIM.",
                "rule_type":          "layer_naming",
                "execution_mode":     "deterministic",
                "severity_default":   "medium",
                "entity_scope":       ["layers"],
                "extraction_confidence": 0.90,
            }
        ]

        conn, cur = make_mock_conn_cur()
        cur.fetchone.side_effect = [None, {"id": 10}]

        with patch("cadsentinel.etl.spec_ingestor.get_connection") as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__  = MagicMock(return_value=False)

            ingestor = SpecIngestor(skip_extraction=True)
            ingestor.skip_extraction = False
            ingestor.client = mock_client

            result = ingestor.ingest(spec_file)

        assert result["rules_extracted"] == 1
        mock_client.extract_specs.assert_called()