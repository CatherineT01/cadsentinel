"""
cadsentinel.etl.evaluation.eval_writer
----------------------------------------
Persists RAGAS evaluation scores to spec_eval_scores.
Also provides a batch evaluator that scores all runs
in a completed spellcheck run.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..db import get_connection
from .ragas_scorer import RAGASScorer, SpecEvalResult

log = logging.getLogger(__name__)


def write_eval_scores(cur, result: SpecEvalResult) -> int:
    """
    Write all metric rows for one SpecEvalResult to spec_eval_scores.
    Returns the number of rows inserted.
    """
    rows    = result.to_metric_rows()
    inserted = 0

    for row in rows:
        try:
            cur.execute(
                """
                INSERT INTO spec_eval_scores (
                    spec_execution_run_id,
                    metric_name,
                    metric_value,
                    evaluator_name
                ) VALUES (%s, %s, %s, %s)
                """,
                (
                    row["spec_execution_run_id"],
                    row["metric_name"],
                    row["metric_value"],
                    row["evaluator_name"],
                ),
            )
            inserted += 1
        except Exception:
            log.exception(
                "Failed to write metric '%s' for run_id=%d",
                row["metric_name"],
                row["spec_execution_run_id"],
            )

    return inserted


def evaluate_spellcheck_run(
    spellcheck_run_id: int,
    use_embeddings:    bool = True,
    gold_labels:       Optional[dict[int, str]] = None,
) -> dict:
    """
    Evaluate all spec execution runs in a completed spellcheck run.
    Loads run data from DB, scores each run, writes scores back.

    Args:
        spellcheck_run_id: ID of the completed spellcheck_runs row
        use_embeddings:    Use embedding similarity for retrieval_relevance
        gold_labels:       Optional dict of {spec_rule_id: pass_fail} gold labels

    Returns:
        dict with counts and average composite score
    """
    scorer = RAGASScorer(use_embeddings=use_embeddings)

    # Load all execution runs for this spellcheck run
    runs = _load_execution_runs(spellcheck_run_id)
    if not runs:
        log.warning(
            "No execution runs found for spellcheck_run_id=%d",
            spellcheck_run_id
        )
        return {
            "spellcheck_run_id": spellcheck_run_id,
            "runs_evaluated":    0,
            "avg_composite":     None,
            "error":             "No execution runs found",
        }

    log.info(
        "Evaluating %d spec runs for spellcheck_run_id=%d",
        len(runs), spellcheck_run_id
    )

    composite_scores = []
    runs_evaluated   = 0

    for run in runs:
        spec_execution_run_id = run["id"]
        spec_rule_id          = run["spec_rule_id"]

        gold_pass_fail = None
        if gold_labels:
            gold_pass_fail = gold_labels.get(spec_rule_id)

        # Reconstruct evidence package from stored JSON
        evidence_package = {
            "evidence": run.get("retrieved_evidence") or [],
        }

        execution_result = run.get("detailed_result") or {}
        if not execution_result:
            execution_result = {
                "pass_fail":     run.get("pass_fail", "needs_review"),
                "confidence":    float(run.get("confidence") or 0),
                "issue_summary": run.get("issue_summary", ""),
                "issues":        [],
                "evidence_used": [],
            }

        try:
            eval_result = scorer.score(
                spec_execution_run_id  = spec_execution_run_id,
                rule_text              = run.get("normalized_rule_text", ""),
                rule_type              = run.get("rule_type", "general"),
                expected_evidence_types = run.get("expected_evidence_types") or [],
                evidence_package       = evidence_package,
                execution_result       = execution_result,
                gold_pass_fail         = gold_pass_fail,
            )

            with get_connection() as conn:
                with conn.cursor() as cur:
                    write_eval_scores(cur, eval_result)
                    conn.commit()

            if eval_result.composite_score is not None:
                composite_scores.append(eval_result.composite_score)
            runs_evaluated += 1

        except Exception:
            log.exception(
                "Failed to evaluate spec_execution_run_id=%d",
                spec_execution_run_id
            )

    avg_composite = (
        round(sum(composite_scores) / len(composite_scores), 4)
        if composite_scores else None
    )

    log.info(
        "Evaluation complete: %d runs, avg_composite=%.4f",
        runs_evaluated, avg_composite or 0,
    )

    return {
        "spellcheck_run_id": spellcheck_run_id,
        "runs_evaluated":    runs_evaluated,
        "avg_composite":     avg_composite,
    }


def _load_execution_runs(spellcheck_run_id: int) -> list[dict]:
    """Load all spec_execution_runs for a spellcheck run, with rule metadata."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    ser.id,
                    ser.spec_rule_id,
                    ser.pass_fail,
                    ser.confidence,
                    ser.issue_summary,
                    ser.detailed_result,
                    ser.retrieved_evidence,
                    sr.normalized_rule_text,
                    sr.rule_type,
                    sr.expected_evidence_types
                FROM spec_execution_runs ser
                JOIN spec_rules sr ON sr.id = ser.spec_rule_id
                WHERE ser.spellcheck_run_id = %s
                ORDER BY ser.id
                """,
                (spellcheck_run_id,),
            )
            return [dict(r) for r in cur.fetchall()]