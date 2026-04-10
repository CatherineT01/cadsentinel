# test_execution.py

"""
tests/test_execution.py
------------------------
Unit tests for the spec execution engine.
No real database, LLM, or DWG file required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

from cadsentinel.etl.execution.llm_checker import (
    format_evidence_for_prompt,
    _parse_llm_result,
    _error_result,
)
from cadsentinel.etl.execution.result_writer import (
    write_execution_result,
    create_spellcheck_run,
    mark_run_complete,
    write_run_summary,
)
from cadsentinel.etl.execution.engine import SpellcheckEngine


# ── Fixtures ─────────────────────────────────────────────────── #

def make_cur():
    cur = MagicMock()
    cur.fetchone.return_value = {"id": 1}
    return cur


def make_conn(cur):
    conn = MagicMock()
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__  = MagicMock(return_value=False)
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
    conn.cursor.return_value.__exit__  = MagicMock(return_value=False)
    return conn


SAMPLE_EVIDENCE = {
    "spec_rule_id":      142,
    "drawing_id":        7,
    "retrieval_method":  "hybrid",
    "query_text":        "All dimensions must be in inches.",
    "evidence_count":    3,
    "truncated":         False,
    "evidence": [
        {
            "source":        "drawing_title_block",
            "found":         True,
            "block_name":    "TITLE_BLOCK",
            "entity_handle": "TB1",
            "attributes":    {"DRAWING_NO": "DWG-001", "UNITS": "INCHES"},
            "geometry":      {},
            "detection_confidence": 0.9,
        },
        {
            "source":          "drawing_dimensions",
            "dim_type":        "DIMENSION_LINEAR",
            "entity_handle":   "D1",
            "layer":           "DIM",
            "measured_value":  4.25,
            "user_text":       None,
            "text_position":   None,
            "geometry":        {},
            "similarity_score": None,
        },
        {
            "source":          "drawing_text_chunks",
            "entity_handle":   "T1",
            "entity_type":     "MTEXT",
            "layer":           "ANNO",
            "text":            "ALL DIMENSIONS IN INCHES UNLESS OTHERWISE NOTED",
            "position":        None,
            "similarity_score": 0.94,
        },
    ],
}

SAMPLE_RULE = {
    "id":                    142,
    "spec_code":             "3.1",
    "spec_title":            "Dimension Units",
    "normalized_rule_text":  "All dimensions must be shown in inches.",
    "rule_type":             "dimension_units",
    "execution_mode":        "deterministic",
    "severity_default":      "medium",
    "entity_scope":          ["dimensions", "notes"],
    "expected_evidence_types": ["dimension", "note"],
    "checker_prompt":        None,
    "retrieval_recipe":      {"source_types": ["dimension", "title_block"], "top_k": 8},
    "structured_rule":       {"required_unit": "inches"},
    "rule_version":          1,
}


# ── format_evidence_for_prompt tests ─────────────────────────── #

class TestFormatEvidenceForPrompt:
    def test_returns_string(self):
        result = format_evidence_for_prompt(SAMPLE_EVIDENCE)
        assert isinstance(result, str)

    def test_includes_title_block_attributes(self):
        result = format_evidence_for_prompt(SAMPLE_EVIDENCE)
        assert "DRAWING_NO" in result or "UNITS" in result

    def test_includes_dimension_info(self):
        result = format_evidence_for_prompt(SAMPLE_EVIDENCE)
        assert "4.25" in result or "DIM" in result

    def test_includes_text_content(self):
        result = format_evidence_for_prompt(SAMPLE_EVIDENCE)
        assert "INCHES" in result

    def test_empty_evidence_returns_message(self):
        result = format_evidence_for_prompt({"evidence": []})
        assert "No evidence available" in result

    def test_handles_missing_evidence_key(self):
        result = format_evidence_for_prompt({})
        assert isinstance(result, str)

    def test_text_similarity_score_included(self):
        result = format_evidence_for_prompt(SAMPLE_EVIDENCE)
        assert "0.94" in result or "relevance" in result.lower()


# ── _parse_llm_result tests ───────────────────────────────────── #

class TestParseLLMResult:
    def test_parses_valid_pass_result(self):
        content = json.dumps({
            "pass_fail":     "pass",
            "confidence":    0.95,
            "severity":      "medium",
            "issue_summary": "",
            "issues":        [],
            "evidence_used": [],
        })
        result = _parse_llm_result(content, "medium")
        assert result["pass_fail"]   == "pass"
        assert result["confidence"]  == 0.95
        assert result["issues"]      == []

    def test_parses_valid_fail_result(self):
        content = json.dumps({
            "pass_fail":     "fail",
            "confidence":    0.88,
            "severity":      "high",
            "issue_summary": "Units wrong",
            "issues":        [{"issue_type": "unit_conflict", "description": "mm found", "evidence": [], "suggested_fix": "Use inches"}],
            "evidence_used": [],
        })
        result = _parse_llm_result(content, "high")
        assert result["pass_fail"]    == "fail"
        assert len(result["issues"]) == 1

    def test_handles_markdown_fences(self):
        content = '```json\n{"pass_fail": "pass", "confidence": 1.0, "severity": "low", "issue_summary": "", "issues": [], "evidence_used": []}\n```'
        result = _parse_llm_result(content, "low")
        assert result["pass_fail"] == "pass"

    def test_defaults_unknown_pass_fail_to_needs_review(self):
        content = json.dumps({
            "pass_fail":     "maybe",
            "confidence":    0.5,
            "severity":      "medium",
            "issue_summary": "",
            "issues":        [],
            "evidence_used": [],
        })
        result = _parse_llm_result(content, "medium")
        assert result["pass_fail"] == "needs_review"

    def test_clamps_confidence_above_1(self):
        content = json.dumps({
            "pass_fail": "pass", "confidence": 1.5,
            "severity": "low", "issue_summary": "",
            "issues": [], "evidence_used": [],
        })
        result = _parse_llm_result(content, "low")
        assert result["confidence"] <= 1.0

    def test_clamps_confidence_below_0(self):
        content = json.dumps({
            "pass_fail": "fail", "confidence": -0.3,
            "severity": "medium", "issue_summary": "x",
            "issues": [], "evidence_used": [],
        })
        result = _parse_llm_result(content, "medium")
        assert result["confidence"] >= 0.0

    def test_returns_needs_review_for_invalid_json(self):
        result = _parse_llm_result("not json at all", "medium")
        assert result["pass_fail"] == "needs_review"

    def test_returns_needs_review_for_empty_content(self):
        result = _parse_llm_result("", "medium")
        assert result["pass_fail"] == "needs_review"

    def test_sanitises_issue_list(self):
        content = json.dumps({
            "pass_fail": "fail", "confidence": 0.8,
            "severity": "medium", "issue_summary": "x",
            "issues": ["not a dict", {"issue_type": "t", "description": "d", "evidence": [], "suggested_fix": ""}],
            "evidence_used": [],
        })
        result = _parse_llm_result(content, "medium")
        # "not a dict" should be filtered out
        assert all(isinstance(i, dict) for i in result["issues"])

    def test_defaults_severity_from_rule_when_invalid(self):
        content = json.dumps({
            "pass_fail": "pass", "confidence": 1.0,
            "severity": "ultra_extreme", "issue_summary": "",
            "issues": [], "evidence_used": [],
        })
        result = _parse_llm_result(content, "high")
        assert result["severity"] == "high"


# ── _error_result tests ───────────────────────────────────────── #

class TestErrorResult:
    def test_returns_needs_review(self):
        result = _error_result("API timeout", "high", "gpt-4o", 5000)
        assert result["pass_fail"]    == "needs_review"
        assert result["confidence"]   == 0.0
        assert result["latency_ms"]   == 5000
        assert result["model_name"]   == "gpt-4o"
        assert len(result["issues"]) == 1
        assert result["issues"][0]["issue_type"] == "llm_error"


# ── result_writer tests ───────────────────────────────────────── #

class TestWriteExecutionResult:
    def test_inserts_execution_run_row(self):
        cur = make_cur()
        result = {
            "pass_fail":     "pass",
            "confidence":    0.97,
            "severity":      "medium",
            "issue_summary": "",
            "issues":        [],
            "evidence_used": [],
        }
        run_id = write_execution_result(
            cur               = cur,
            spellcheck_run_id = 1,
            drawing_id        = 7,
            spec_rule_id      = 142,
            rule_version      = 1,
            execution_mode    = "deterministic",
            result            = result,
            evidence_package  = SAMPLE_EVIDENCE,
        )
        assert run_id == 1
        assert cur.execute.called

    def test_writes_issues_when_fail(self):
        cur = make_cur()
        cur.fetchone.return_value = {"id": 55}
        result = {
            "pass_fail":     "fail",
            "confidence":    0.90,
            "severity":      "high",
            "issue_summary": "Missing field",
            "issues": [
                {"issue_type": "title_block_field_missing", "severity": "high",
                 "description": "REVISION missing", "suggested_fix": "Add REVISION",
                 "entity_ref": None, "confidence": 1.0},
            ],
            "evidence_used": [],
        }
        write_execution_result(
            cur=cur, spellcheck_run_id=1, drawing_id=7,
            spec_rule_id=142, rule_version=1,
            execution_mode="deterministic",
            result=result, evidence_package=SAMPLE_EVIDENCE,
        )
        # execute called at least twice: once for run, once per issue
        assert cur.execute.call_count >= 2

    def test_returns_none_on_db_error(self):
        cur = make_cur()
        cur.execute.side_effect = Exception("DB error")
        result = {"pass_fail": "pass", "confidence": 1.0, "severity": "low",
                  "issue_summary": "", "issues": [], "evidence_used": []}
        run_id = write_execution_result(
            cur=cur, spellcheck_run_id=1, drawing_id=7,
            spec_rule_id=142, rule_version=1,
            execution_mode="deterministic",
            result=result, evidence_package=SAMPLE_EVIDENCE,
        )
        assert run_id is None


class TestCreateSpellcheckRun:
    def test_inserts_run_row_and_returns_id(self):
        cur = make_cur()
        cur.fetchone.return_value = {"id": 99}
        run_id = create_spellcheck_run(
            cur=cur, drawing_id=7, spec_document_id=3,
            total_specs=10, triggered_by="test",
        )
        assert run_id == 99
        assert cur.execute.called

    def test_sql_contains_running_status(self):
        cur = make_cur()
        cur.fetchone.return_value = {"id": 1}
        create_spellcheck_run(
            cur=cur, drawing_id=7, spec_document_id=3, total_specs=5,
        )
        sql = cur.execute.call_args[0][0]
        assert "running" in sql


class TestMarkRunComplete:
    def test_updates_run_status(self):
        cur = make_cur()
        mark_run_complete(cur, spellcheck_run_id=99, specs_completed=10)
        sql = cur.execute.call_args[0][0]
        assert "completed" in sql
        assert 99 in cur.execute.call_args[0][1]


# ── SpellcheckEngine tests ────────────────────────────────────── #

class TestSpellcheckEngine:
    def _make_engine(self):
        return SpellcheckEngine(
            provider            = "openai",
            embed_for_retrieval = False,
            max_workers         = 2,
        )

    def test_returns_empty_result_when_no_approved_rules(self):
        engine = self._make_engine()
        cur    = make_cur()
        cur.fetchall.return_value = []
        cur.fetchone.return_value = None
        conn   = make_conn(cur)

        with patch("cadsentinel.etl.execution.engine.get_connection") as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__  = MagicMock(return_value=False)

            result = engine.run(drawing_id=7, spec_document_id=3)

        assert result["total_specs"] == 0
        assert "error" in result

    def test_route_deterministic_calls_validator(self):
        engine = self._make_engine()
        result = engine._route_and_execute(
            execution_mode   = "deterministic",
            rule_type        = "dimension_units",
            rule_config      = {"required_unit": "inches", "severity_default": "medium"},
            rule             = SAMPLE_RULE,
            evidence_package = SAMPLE_EVIDENCE,
        )
        assert "pass_fail" in result
        assert result["pass_fail"] in ("pass", "fail", "warning", "needs_review")

    def test_route_unknown_mode_returns_needs_review(self):
        engine = self._make_engine()
        result = engine._route_and_execute(
            execution_mode   = "deterministic",
            rule_type        = "completely_unknown_type_xyz",
            rule_config      = {"severity_default": "medium"},
            rule             = {**SAMPLE_RULE, "normalized_rule_text": "Some rule."},
            evidence_package = SAMPLE_EVIDENCE,
        )
        assert result["pass_fail"] == "needs_review"

    def test_deterministic_dimension_units_pass(self):
        engine = self._make_engine()
        result = engine._run_deterministic(
            rule_type        = "dimension_units",
            rule_config      = {"required_unit": "inches", "severity_default": "medium"},
            evidence_package = SAMPLE_EVIDENCE,
        )
        assert result["pass_fail"] == "pass"

    def test_deterministic_title_block_fail_missing_field(self):
        engine = self._make_engine()
        evidence = {
            "evidence": [{
                "source":        "drawing_title_block",
                "found":         True,
                "entity_handle": "TB1",
                "attributes":    {"DRAWING_NO": "X"},
                "geometry":      {},
                "detection_confidence": 0.9,
            }]
        }
        result = engine._run_deterministic(
            rule_type        = "title_block",
            rule_config      = {"required_fields": ["DRAWING_NO", "REVISION"], "severity_default": "high"},
            evidence_package = evidence,
        )
        assert result["pass_fail"] == "fail"

    def test_run_summary_has_required_keys(self):
        engine = self._make_engine()

        mock_rules = [dict(SAMPLE_RULE)]

        cur  = make_cur()
        cur.fetchone.side_effect = [
            {"id": 5},    # create_spellcheck_run
            None,         # load_approved_rules handled separately
        ]
        cur.fetchall.return_value = [dict(SAMPLE_RULE)]
        conn = make_conn(cur)

        with patch("cadsentinel.etl.execution.engine.get_connection") as mock_gc, \
             patch.object(engine, "_load_approved_rules", return_value=mock_rules), \
             patch.object(engine, "_execute_parallel", return_value=[
                 {"pass_fail": "pass", "confidence": 0.97}
             ]):

            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__  = MagicMock(return_value=False)

            result = engine.run(drawing_id=7, spec_document_id=3)

        assert "spellcheck_run_id" in result
        assert "pass_count"        in result
        assert "fail_count"        in result
        assert "warning_count"     in result
        assert "review_count"      in result
        assert "total_specs"       in result