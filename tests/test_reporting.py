# test_reporting.py

"""
tests/test_reporting.py
------------------------
Unit tests for the reporting layer and approval CLI.
No real database or FastAPI server required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cadsentinel.etl.reporting.report_generator import ReportGenerator


# ── Fixtures ─────────────────────────────────────────────────── #

SAMPLE_RUN = {
    "id":               99,
    "drawing_id":       7,
    "spec_document_id": 3,
    "run_status":       "completed",
    "total_specs":      10,
    "specs_completed":  10,
    "started_at":       None,
    "completed_at":     None,
    "model_family":     "openai",
}

SAMPLE_SPEC_RESULTS = [
    {"id": 1, "spec_rule_id": 101, "execution_mode": "deterministic",
     "pass_fail": "pass", "severity": "medium", "confidence": 0.97,
     "issue_count": 0, "issue_summary": "", "latency_ms": 12,
     "spec_code": "3.1", "spec_title": "Layer Naming",
     "rule_type": "layer_naming", "normalized_rule_text": "Layers must be named..."},
    {"id": 2, "spec_rule_id": 102, "execution_mode": "deterministic",
     "pass_fail": "fail", "severity": "high", "confidence": 1.0,
     "issue_count": 1, "issue_summary": "REVISION field missing",
     "latency_ms": 8, "spec_code": "3.2", "spec_title": "Title Block",
     "rule_type": "title_block", "normalized_rule_text": "Title block must..."},
    {"id": 3, "spec_rule_id": 103, "execution_mode": "hybrid",
     "pass_fail": "warning", "severity": "low", "confidence": 0.75,
     "issue_count": 1, "issue_summary": "Units not declared",
     "latency_ms": 340, "spec_code": "3.3", "spec_title": "Dimension Units",
     "rule_type": "dimension_units", "normalized_rule_text": "All dims in inches..."},
    {"id": 4, "spec_rule_id": 104, "execution_mode": "llm_judge",
     "pass_fail": "needs_review", "severity": "medium", "confidence": 0.0,
     "issue_count": 0, "issue_summary": "Insufficient evidence",
     "latency_ms": 890, "spec_code": "3.4", "spec_title": "Safety Note",
     "rule_type": "safety_note", "normalized_rule_text": "Safety note must..."},
]

SAMPLE_RAGAS = [
    {"metric_name": "composite_score",       "avg_value": 0.84, "n": 4},
    {"metric_name": "retrieval_relevance",   "avg_value": 0.78, "n": 4},
    {"metric_name": "faithfulness",          "avg_value": 0.90, "n": 4},
    {"metric_name": "false_positive_risk",   "avg_value": 0.12, "n": 4},
]


def make_generator_with_mocks(run=None, spec_results=None, ragas=None):
    """Return a ReportGenerator with all DB methods mocked."""
    gen = ReportGenerator()
    gen._load_run          = MagicMock(return_value=run or SAMPLE_RUN)
    gen._load_spec_results = MagicMock(return_value=spec_results or SAMPLE_SPEC_RESULTS)
    gen._load_ragas_scores = MagicMock(return_value=ragas or SAMPLE_RAGAS)
    return gen


# ── Report structure tests ────────────────────────────────────── #

class TestReportStructure:
    def test_report_has_required_keys(self):
        gen    = make_generator_with_mocks()
        report = gen.generate(99)
        for key in [
            "spellcheck_run_id", "drawing_id", "spec_document_id",
            "generated_at", "overall_grade", "pass_rate", "summary",
            "counts", "severity_fails", "spec_results",
            "ragas_summary", "recommendations",
        ]:
            assert key in report, f"Missing key: {key}"

    def test_report_error_when_run_not_found(self):
        gen = ReportGenerator()
        gen._load_run = MagicMock(return_value=None)
        report = gen.generate(999)
        assert "error" in report

    def test_counts_match_spec_results(self):
        gen    = make_generator_with_mocks()
        report = gen.generate(99)
        assert report["counts"]["pass"]         == 1
        assert report["counts"]["fail"]         == 1
        assert report["counts"]["warning"]      == 1
        assert report["counts"]["needs_review"] == 1

    def test_pass_rate_correct(self):
        gen    = make_generator_with_mocks()
        report = gen.generate(99)
        assert report["pass_rate"] == 0.25   # 1 pass / 4 total

    def test_spec_results_included(self):
        gen    = make_generator_with_mocks()
        report = gen.generate(99)
        assert len(report["spec_results"]) == 4

    def test_ragas_summary_included(self):
        gen    = make_generator_with_mocks()
        report = gen.generate(99)
        assert "composite_score"     in report["ragas_summary"]
        assert "retrieval_relevance" in report["ragas_summary"]

    def test_recommendations_is_list(self):
        gen    = make_generator_with_mocks()
        report = gen.generate(99)
        assert isinstance(report["recommendations"], list)
        assert len(report["recommendations"]) > 0

    def test_generated_at_is_string(self):
        gen    = make_generator_with_mocks()
        report = gen.generate(99)
        assert isinstance(report["generated_at"], str)
        assert "T" in report["generated_at"]   # ISO format


# ── Grade computation tests ───────────────────────────────────── #

class TestGradeComputation:
    gen = ReportGenerator()

    def test_grade_A_at_high_pass_rate(self):
        grade = self.gen._grade_from_rate(0.97)
        assert grade == "A"

    def test_grade_B(self):
        grade = self.gen._grade_from_rate(0.87)
        assert grade == "B"

    def test_grade_C(self):
        grade = self.gen._grade_from_rate(0.72)
        assert grade == "C"

    def test_grade_D(self):
        grade = self.gen._grade_from_rate(0.55)
        assert grade == "D"

    def test_grade_F_at_low_pass_rate(self):
        grade = self.gen._grade_from_rate(0.30)
        assert grade == "F"

    def test_critical_failure_caps_grade_at_D(self):
        grade = self.gen._compute_grade(
            pass_rate      = 0.90,
            severity_fails = {"critical": 1, "high": 0, "medium": 0, "low": 0},
            counts         = {"pass": 9, "fail": 1},
        )
        assert grade in ("D", "F")

    def test_many_high_failures_cap_grade_at_C(self):
        grade = self.gen._compute_grade(
            pass_rate      = 0.92,
            severity_fails = {"critical": 0, "high": 5, "medium": 0, "low": 0},
            counts         = {"pass": 92, "fail": 8},
        )
        assert grade in ("C", "D", "F")

    def test_no_failures_produces_A(self):
        grade = self.gen._compute_grade(
            pass_rate      = 1.0,
            severity_fails = {"critical": 0, "high": 0, "medium": 0, "low": 0},
            counts         = {"pass": 10, "fail": 0},
        )
        assert grade == "A"


# ── Summary text tests ────────────────────────────────────────── #

class TestSummaryText:
    gen = ReportGenerator()

    def test_summary_contains_total_count(self):
        summary = self.gen._build_summary(
            SAMPLE_RUN,
            {"pass": 8, "fail": 1, "warning": 1, "needs_review": 0},
            10, 0.80, "B"
        )
        assert "10" in summary

    def test_summary_contains_grade(self):
        summary = self.gen._build_summary(
            SAMPLE_RUN,
            {"pass": 9, "fail": 0, "warning": 1, "needs_review": 0},
            10, 0.90, "B"
        )
        assert "B" in summary

    def test_summary_no_issues_message(self):
        summary = self.gen._build_summary(
            SAMPLE_RUN,
            {"pass": 10, "fail": 0, "warning": 0, "needs_review": 0},
            10, 1.0, "A"
        )
        assert "No compliance issues" in summary

    def test_summary_fail_requires_attention(self):
        summary = self.gen._build_summary(
            SAMPLE_RUN,
            {"pass": 8, "fail": 2, "warning": 0, "needs_review": 0},
            10, 0.80, "B"
        )
        assert "attention" in summary.lower() or "require" in summary.lower()


# ── Recommendation tests ──────────────────────────────────────── #

class TestRecommendations:
    gen = ReportGenerator()

    def test_critical_failure_triggers_critical_rec(self):
        recs = self.gen._build_recommendations(
            {"pass": 9, "fail": 1},
            {"critical": 1, "high": 0, "medium": 0, "low": 0},
            {},
        )
        assert any("CRITICAL" in r for r in recs)

    def test_high_severity_triggers_review_rec(self):
        recs = self.gen._build_recommendations(
            {"pass": 8, "fail": 2},
            {"critical": 0, "high": 2, "medium": 0, "low": 0},
            {},
        )
        assert any("high-severity" in r.lower() or "engineering review" in r.lower() for r in recs)

    def test_needs_review_triggers_recipe_rec(self):
        recs = self.gen._build_recommendations(
            {"pass": 8, "fail": 0, "warning": 0, "needs_review": 2},
            {"critical": 0, "high": 0, "medium": 0, "low": 0},
            {},
        )
        assert any("needs_review" in r or "retrieval" in r.lower() for r in recs)

    def test_low_retrieval_relevance_triggers_recipe_rec(self):
        recs = self.gen._build_recommendations(
            {"pass": 10}, {}, {"retrieval_relevance": 0.40}
        )
        assert any("retrieval" in r.lower() for r in recs)

    def test_high_fpr_triggers_hallucination_warning(self):
        recs = self.gen._build_recommendations(
            {"pass": 10}, {}, {"false_positive_risk": 0.70}
        )
        assert any("false positive" in r.lower() or "hallucination" in r.lower() for r in recs)

    def test_all_pass_gives_approval_rec(self):
        recs = self.gen._build_recommendations(
            {"pass": 10, "fail": 0, "warning": 0, "needs_review": 0},
            {"critical": 0, "high": 0, "medium": 0, "low": 0},
            {},
        )
        assert any("approval" in r.lower() or "meets all" in r.lower() for r in recs)


# ── RAGAS summary tests ───────────────────────────────────────── #

class TestRAGASSummary:
    gen = ReportGenerator()

    def test_summarise_ragas_extracts_values(self):
        rows = [
            {"metric_name": "retrieval_relevance", "avg_value": 0.85, "n": 5},
            {"metric_name": "faithfulness",        "avg_value": 0.90, "n": 5},
        ]
        summary = self.gen._summarise_ragas(rows)
        assert abs(summary["retrieval_relevance"] - 0.85) < 0.001
        assert abs(summary["faithfulness"] - 0.90) < 0.001

    def test_summarise_ragas_skips_none(self):
        rows = [{"metric_name": "stability", "avg_value": None, "n": 0}]
        summary = self.gen._summarise_ragas(rows)
        assert "stability" not in summary

    def test_summarise_ragas_empty_input(self):
        assert self.gen._summarise_ragas([]) == {}


# ── Full report integration ───────────────────────────────────── #

class TestFullReport:
    def test_all_pass_report_grades_A(self):
        all_pass = [
            {**r, "pass_fail": "pass", "severity": "medium", "issue_count": 0}
            for r in SAMPLE_SPEC_RESULTS
        ]
        gen    = make_generator_with_mocks(spec_results=all_pass)
        report = gen.generate(99)
        assert report["overall_grade"] == "A"
        assert report["pass_rate"]     == 1.0

    def test_all_fail_report_grades_F(self):
        all_fail = [
            {**r, "pass_fail": "fail", "severity": "medium"}
            for r in SAMPLE_SPEC_RESULTS
        ]
        gen    = make_generator_with_mocks(spec_results=all_fail)
        report = gen.generate(99)
        assert report["overall_grade"] in ("D", "F")

    def test_empty_spec_results_handled(self):
        gen = ReportGenerator()
        gen._load_run          = MagicMock(return_value=SAMPLE_RUN)
        gen._load_spec_results = MagicMock(return_value=[])
        gen._load_ragas_scores = MagicMock(return_value=[])
        report = gen.generate(99)
        assert report["pass_rate"] == 0.0
        assert report["overall_grade"] == "F"

    def test_report_drawing_id_matches_run(self):
        gen    = make_generator_with_mocks()
        report = gen.generate(99)
        assert report["drawing_id"] == SAMPLE_RUN["drawing_id"]