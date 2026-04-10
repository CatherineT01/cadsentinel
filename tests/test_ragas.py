# test_ragas.py

"""
tests/test_ragas.py
--------------------
Unit tests for the RAGAS evaluation layer.
No real database, embeddings, or LLM required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cadsentinel.etl.evaluation.ragas_scorer import (
    RAGASScorer,
    SpecEvalResult,
    _tokenize,
    _cosine_similarity,
    _mean_vector,
    _extract_text_from_evidence,
)
from cadsentinel.etl.evaluation.eval_writer import write_eval_scores


# ── Fixtures ─────────────────────────────────────────────────── #

RULE_TEXT = "All dimensions must be shown in inches unless otherwise specified."

EVIDENCE_PACKAGE = {
    "evidence": [
        {
            "source":        "drawing_title_block",
            "found":         True,
            "entity_handle": "TB1",
            "attributes":    {"DRAWING_NO": "DWG-001", "UNITS": "INCHES"},
        },
        {
            "source":          "drawing_dimensions",
            "entity_handle":   "D1",
            "dim_type":        "DIMENSION_LINEAR",
            "measured_value":  4.25,
            "user_text":       None,
            "layer":           "DIM",
        },
        {
            "source":          "drawing_text_chunks",
            "entity_handle":   "T1",
            "text":            "ALL DIMENSIONS IN INCHES UNLESS OTHERWISE NOTED",
            "similarity_score": 0.94,
        },
    ]
}

PASS_RESULT = {
    "pass_fail":     "pass",
    "confidence":    0.97,
    "severity":      "medium",
    "issue_summary": "",
    "issues":        [],
    "evidence_used": [],
}

FAIL_RESULT = {
    "pass_fail":     "fail",
    "confidence":    0.88,
    "severity":      "high",
    "issue_summary": "Dimensions appear to use mm instead of inches.",
    "issues": [{
        "issue_type":    "unit_conflict",
        "description":   "Dimension entities reference mm units.",
        "evidence":      ["Dimension note on sheet references mm"],
        "suggested_fix": "Convert dimensions to inches.",
    }],
    "evidence_used": [{"type": "dimension", "ref": "D1"}],
}


# ── SpecEvalResult tests ───────────────────────────────────────── #

class TestSpecEvalResult:
    def test_to_metric_rows_excludes_none_values(self):
        result = SpecEvalResult(
            spec_execution_run_id = 1,
            retrieval_relevance   = 0.85,
            evidence_coverage     = 0.90,
            decision_correctness  = None,   # no gold label
            faithfulness          = 0.88,
        )
        rows = result.to_metric_rows()
        metric_names = [r["metric_name"] for r in rows]
        assert "retrieval_relevance" in metric_names
        assert "evidence_coverage"   in metric_names
        assert "faithfulness"        in metric_names
        assert "decision_correctness" not in metric_names

    def test_to_metric_rows_includes_composite(self):
        result = SpecEvalResult(
            spec_execution_run_id = 1,
            composite_score       = 0.88,
        )
        rows = result.to_metric_rows()
        assert any(r["metric_name"] == "composite_score" for r in rows)

    def test_to_metric_rows_all_none_returns_empty(self):
        result = SpecEvalResult(spec_execution_run_id=1)
        rows = result.to_metric_rows()
        assert rows == []

    def test_evaluator_name_on_all_rows(self):
        result = SpecEvalResult(
            spec_execution_run_id = 1,
            retrieval_relevance   = 0.80,
            composite_score       = 0.80,
        )
        rows = result.to_metric_rows()
        for row in rows:
            assert row["evaluator_name"] == "cadsentinel_ragas_v1"


# ── RAGASScorer — retrieval_relevance ────────────────────────── #

class TestRetrievalRelevance:
    scorer = RAGASScorer(use_embeddings=False)

    def test_returns_float_between_0_and_1(self):
        score = self.scorer._score_retrieval_relevance(RULE_TEXT, ["inches required"])
        assert 0.0 <= score <= 1.0

    def test_returns_zero_for_empty_evidence(self):
        score = self.scorer._score_retrieval_relevance(RULE_TEXT, [])
        assert score == 0.0

    def test_higher_score_for_relevant_evidence(self):
        relevant = self.scorer._score_retrieval_relevance(
            "dimension units inches", ["all dimensions must be in inches"]
        )
        irrelevant = self.scorer._score_retrieval_relevance(
            "dimension units inches", ["title block drawing number revision"]
        )
        assert relevant > irrelevant

    def test_perfect_overlap_near_1(self):
        score = self.scorer._score_retrieval_relevance(
            "dimensions inches required", ["dimensions inches required"]
        )
        assert score > 0.80

    def test_no_overlap_near_0(self):
        score = self.scorer._score_retrieval_relevance(
            "dimension units", ["layer name prefix border annotation"]
        )
        assert score < 0.30


# ── RAGASScorer — evidence_coverage ─────────────────────────── #

class TestEvidenceCoverage:
    scorer = RAGASScorer(use_embeddings=False)

    def test_full_coverage_when_all_types_present(self):
        items = [
            {"source": "drawing_dimensions"},
            {"source": "drawing_title_block"},
        ]
        score = self.scorer._score_evidence_coverage(
            ["dimension", "title_block"], items
        )
        assert score == 1.0

    def test_partial_coverage(self):
        items = [{"source": "drawing_dimensions"}]
        score = self.scorer._score_evidence_coverage(
            ["dimension", "title_block", "note"], items
        )
        assert 0.0 < score < 1.0

    def test_zero_coverage_when_wrong_types(self):
        items = [{"source": "drawing_layers"}]
        score = self.scorer._score_evidence_coverage(
            ["dimension", "title_block"], items
        )
        assert score == 0.0

    def test_full_coverage_when_no_expectations(self):
        score = self.scorer._score_evidence_coverage([], [])
        assert score == 1.0

    def test_note_maps_to_text_also(self):
        items = [{"source": "drawing_text_chunks"}]
        score = self.scorer._score_evidence_coverage(["text"], items)
        assert score == 1.0


# ── RAGASScorer — decision_correctness ───────────────────────── #

class TestDecisionCorrectness:
    scorer = RAGASScorer(use_embeddings=False)

    def test_exact_match_is_1(self):
        assert self.scorer._score_decision_correctness("pass", "pass") == 1.0
        assert self.scorer._score_decision_correctness("fail", "fail") == 1.0

    def test_wrong_verdict_is_0(self):
        assert self.scorer._score_decision_correctness("pass", "fail") == 0.0

    def test_warning_vs_fail_is_partial(self):
        score = self.scorer._score_decision_correctness("warning", "fail")
        assert 0.0 < score < 1.0

    def test_needs_review_gets_partial_credit(self):
        score = self.scorer._score_decision_correctness("needs_review", "fail")
        assert 0.0 < score < 1.0


# ── RAGASScorer — faithfulness ────────────────────────────────── #

class TestFaithfulness:
    scorer = RAGASScorer(use_embeddings=False)

    def test_pass_result_is_always_faithful(self):
        score = self.scorer._score_faithfulness("", [], EVIDENCE_PACKAGE["evidence"], RULE_TEXT)
        assert score == 1.0

    def test_issue_with_cited_evidence_is_faithful(self):
        issues = [{"description": "mm found", "evidence": ["Dimension note references mm"], "suggested_fix": ""}]
        score  = self.scorer._score_faithfulness("mm found", issues, EVIDENCE_PACKAGE["evidence"], RULE_TEXT)
        assert score >= 0.8

    def test_issue_with_no_evidence_overlap_lower_score(self):
        items  = [{"source": "drawing_text_chunks", "text": "BORDER LAYER REQUIRED"}]
        issues = [{"description": "completely unrelated claim about planets", "evidence": [], "suggested_fix": ""}]
        score  = self.scorer._score_faithfulness("unrelated", issues, items, RULE_TEXT)
        assert score < 0.8

    def test_no_evidence_returns_half(self):
        issues = [{"description": "something wrong", "evidence": [], "suggested_fix": ""}]
        score  = self.scorer._score_faithfulness("issue", issues, [], RULE_TEXT)
        assert score == 0.5


# ── RAGASScorer — false_positive_risk ────────────────────────── #

class TestFalsePositiveRisk:
    scorer = RAGASScorer(use_embeddings=False)

    def test_pass_result_has_zero_risk(self):
        score = self.scorer._score_false_positive_risk("pass", [], EVIDENCE_PACKAGE["evidence"])
        assert score == 0.0

    def test_cited_evidence_lowers_risk(self):
        issues = [{"description": "mm found", "evidence": ["dimension note references mm"], "suggested_fix": ""}]
        score  = self.scorer._score_false_positive_risk("fail", issues, EVIDENCE_PACKAGE["evidence"])
        assert score < 0.20

    def test_no_evidence_raises_risk(self):
        issues = [{"description": "something bad", "evidence": [], "suggested_fix": ""}]
        score  = self.scorer._score_false_positive_risk("fail", issues, [])
        assert score > 0.50

    def test_high_overlap_lowers_risk(self):
        items  = [{"source": "drawing_text_chunks", "text": "dimensions must be in inches only"}]
        issues = [{"description": "dimensions not in inches", "evidence": [], "suggested_fix": ""}]
        score  = self.scorer._score_false_positive_risk("fail", issues, items)
        assert score < 0.50


# ── RAGASScorer — confidence_calibration ─────────────────────── #

class TestConfidenceCalibration:
    scorer = RAGASScorer(use_embeddings=False)

    def test_perfect_calibration(self):
        score = self.scorer._score_confidence_calibration(1.0, 1.0)
        assert score == 1.0

    def test_worst_calibration(self):
        score = self.scorer._score_confidence_calibration(1.0, 0.0)
        assert score == 0.0

    def test_partial_calibration(self):
        score = self.scorer._score_confidence_calibration(0.8, 1.0)
        assert abs(score - 0.8) < 0.01


# ── RAGASScorer — composite score ────────────────────────────── #

class TestCompositeScore:
    scorer = RAGASScorer(use_embeddings=False)

    def test_composite_between_0_and_1(self):
        result = SpecEvalResult(
            spec_execution_run_id = 1,
            retrieval_relevance   = 0.85,
            evidence_coverage     = 0.90,
            faithfulness          = 0.88,
            false_positive_risk   = 0.10,
        )
        score = self.scorer._compute_composite(result)
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_all_none_returns_none(self):
        result = SpecEvalResult(spec_execution_run_id=1)
        assert self.scorer._compute_composite(result) is None

    def test_high_scores_produce_high_composite(self):
        result = SpecEvalResult(
            spec_execution_run_id = 1,
            retrieval_relevance   = 0.95,
            evidence_coverage     = 0.95,
            faithfulness          = 0.95,
            false_positive_risk   = 0.05,
        )
        score = self.scorer._compute_composite(result)
        assert score > 0.80

    def test_low_scores_produce_low_composite(self):
        result = SpecEvalResult(
            spec_execution_run_id = 1,
            retrieval_relevance   = 0.20,
            evidence_coverage     = 0.20,
            faithfulness          = 0.20,
            false_positive_risk   = 0.80,
        )
        score = self.scorer._compute_composite(result)
        assert score < 0.40


# ── RAGASScorer — recommendation ─────────────────────────────── #

class TestRecommendation:
    scorer = RAGASScorer(use_embeddings=False)

    def test_high_score_retains_prompt(self):
        result = SpecEvalResult(
            spec_execution_run_id=1,
            retrieval_relevance=0.92, evidence_coverage=0.90,
            faithfulness=0.91, false_positive_risk=0.05, composite_score=0.90,
        )
        rec = self.scorer._make_recommendation(result)
        assert rec == "retain_current_prompt"

    def test_low_retrieval_recommends_recipe_fix(self):
        result = SpecEvalResult(
            spec_execution_run_id=1,
            retrieval_relevance=0.30, evidence_coverage=0.80,
            faithfulness=0.80, false_positive_risk=0.10, composite_score=0.68,
        )
        rec = self.scorer._make_recommendation(result)
        assert rec == "refine_retrieval_recipe"

    def test_none_composite_returns_insufficient_data(self):
        result = SpecEvalResult(spec_execution_run_id=1)
        rec = self.scorer._make_recommendation(result)
        assert rec == "insufficient_data"


# ── Full score() integration test ────────────────────────────── #

class TestFullScore:
    def test_score_pass_result_no_gold(self):
        scorer = RAGASScorer(use_embeddings=False)
        result = scorer.score(
            spec_execution_run_id   = 99,
            rule_text               = RULE_TEXT,
            rule_type               = "dimension_units",
            expected_evidence_types = ["dimension", "title_block"],
            evidence_package        = EVIDENCE_PACKAGE,
            execution_result        = PASS_RESULT,
            gold_pass_fail          = None,
        )
        assert result.spec_execution_run_id == 99
        assert result.retrieval_relevance   is not None
        assert result.evidence_coverage     is not None
        assert result.faithfulness          is not None
        assert result.false_positive_risk   is not None
        assert result.decision_correctness  is None   # no gold label
        assert result.composite_score       is not None
        assert 0.0 <= result.composite_score <= 1.0

    def test_score_fail_result_with_gold(self):
        scorer = RAGASScorer(use_embeddings=False)
        result = scorer.score(
            spec_execution_run_id   = 100,
            rule_text               = RULE_TEXT,
            rule_type               = "dimension_units",
            expected_evidence_types = ["dimension"],
            evidence_package        = EVIDENCE_PACKAGE,
            execution_result        = FAIL_RESULT,
            gold_pass_fail          = "fail",
        )
        assert result.decision_correctness  == 1.0
        assert result.confidence_calibration is not None
        assert result.composite_score        is not None

    def test_score_wrong_verdict_lowers_correctness(self):
        scorer = RAGASScorer(use_embeddings=False)
        result = scorer.score(
            spec_execution_run_id   = 101,
            rule_text               = RULE_TEXT,
            rule_type               = "dimension_units",
            expected_evidence_types = ["dimension"],
            evidence_package        = EVIDENCE_PACKAGE,
            execution_result        = PASS_RESULT,   # predicted pass
            gold_pass_fail          = "fail",        # actually fail
        )
        assert result.decision_correctness == 0.0

    def test_metric_rows_written_correctly(self):
        scorer = RAGASScorer(use_embeddings=False)
        result = scorer.score(
            spec_execution_run_id   = 102,
            rule_text               = RULE_TEXT,
            rule_type               = "dimension_units",
            expected_evidence_types = ["dimension"],
            evidence_package        = EVIDENCE_PACKAGE,
            execution_result        = PASS_RESULT,
        )
        rows = result.to_metric_rows()
        assert len(rows) > 0
        assert all("metric_name" in r for r in rows)
        assert all("metric_value" in r for r in rows)


# ── Utility function tests ────────────────────────────────────── #

class TestUtilities:
    def test_tokenize_lowercases(self):
        assert _tokenize("INCHES Dimensions") == ["inches", "dimensions"]

    def test_tokenize_strips_punctuation(self):
        assert _tokenize("inch.es, req'd") == ["inch", "es", "req", "d"]

    def test_cosine_similarity_identical(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 0.001

    def test_mean_vector(self):
        vecs = [[1.0, 2.0], [3.0, 4.0]]
        mean = _mean_vector(vecs)
        assert abs(mean[0] - 2.0) < 0.001
        assert abs(mean[1] - 3.0) < 0.001

    def test_extract_text_from_title_block(self):
        items = [{"source": "drawing_title_block", "attributes": {"UNITS": "INCHES", "DRAWING_NO": "X"}}]
        texts = _extract_text_from_evidence(items)
        assert any("INCHES" in t for t in texts)

    def test_extract_text_from_chunks(self):
        items = [{"source": "drawing_text_chunks", "text": "ALL DIMENSIONS IN INCHES"}]
        texts = _extract_text_from_evidence(items)
        assert "ALL DIMENSIONS IN INCHES" in texts

    def test_extract_text_from_layers(self):
        items = [{"source": "drawing_layers", "layer_name": "DIM-LINEAR"}]
        texts = _extract_text_from_evidence(items)
        assert "DIM-LINEAR" in texts


# ── eval_writer tests ─────────────────────────────────────────── #

class TestWriteEvalScores:
    def test_inserts_correct_number_of_rows(self):
        cur = MagicMock()
        result = SpecEvalResult(
            spec_execution_run_id = 1,
            retrieval_relevance   = 0.85,
            evidence_coverage     = 0.90,
            faithfulness          = 0.88,
            composite_score       = 0.87,
        )
        inserted = write_eval_scores(cur, result)
        assert inserted == 4
        assert cur.execute.call_count == 4

    def test_skips_none_metrics(self):
        cur = MagicMock()
        result = SpecEvalResult(
            spec_execution_run_id = 1,
            retrieval_relevance   = 0.85,
            # all others None
        )
        inserted = write_eval_scores(cur, result)
        assert inserted == 1

    def test_returns_zero_on_db_error(self):
        cur = MagicMock()
        cur.execute.side_effect = Exception("DB error")
        result = SpecEvalResult(
            spec_execution_run_id=1,
            retrieval_relevance=0.85,
        )
        inserted = write_eval_scores(cur, result)
        assert inserted == 0