"""
cadsentinel.etl.reporting.report_generator
-------------------------------------------
Generates structured compliance reports from completed spellcheck runs.

A compliance report contains:
  - overall grade (A/B/C/D/F based on pass rate and severity)
  - per-spec results grouped by severity
  - issue summaries
  - RAGAS scores where available
  - recommendations

Usage:
    generator = ReportGenerator()
    report    = generator.generate(spellcheck_run_id=99)
    print(report["overall_grade"], report["summary"])
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from ..db import get_connection

log = logging.getLogger(__name__)

# Grade thresholds based on pass rate and severity mix
GRADE_THRESHOLDS = {
    "A": 0.95,   # >= 95% pass
    "B": 0.85,   # >= 85% pass
    "C": 0.70,   # >= 70% pass
    "D": 0.50,   # >= 50% pass
    # Below 50% = F
}


class ReportGenerator:
    """Generates compliance reports from completed spellcheck runs."""

    def generate(self, spellcheck_run_id: int) -> dict:
        """
        Generate a full compliance report for a spellcheck run.

        Returns:
            dict with overall_grade, summary, spec_results,
                  severity_counts, ragas_summary, recommendations
        """
        run = self._load_run(spellcheck_run_id)
        if not run:
            return {
                "error": f"Spellcheck run {spellcheck_run_id} not found",
                "spellcheck_run_id": spellcheck_run_id,
            }

        spec_results  = self._load_spec_results(spellcheck_run_id)
        ragas_scores  = self._load_ragas_scores(spellcheck_run_id)

        # Tally verdicts
        counts = {"pass": 0, "fail": 0, "warning": 0, "needs_review": 0}
        severity_fails = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        for spec in spec_results:
            verdict = spec.get("pass_fail", "needs_review")
            counts[verdict] = counts.get(verdict, 0) + 1
            if verdict in ("fail", "warning"):
                sev = spec.get("severity", "medium")
                severity_fails[sev] = severity_fails.get(sev, 0) + 1

        total = sum(counts.values())
        pass_rate = counts["pass"] / total if total > 0 else 0.0

        overall_grade = self._compute_grade(
            pass_rate, severity_fails, counts
        )

        summary = self._build_summary(
            run, counts, total, pass_rate, overall_grade
        )

        ragas_summary = self._summarise_ragas(ragas_scores)
        recommendations = self._build_recommendations(
            counts, severity_fails, ragas_summary
        )

        return {
            "spellcheck_run_id":  spellcheck_run_id,
            "drawing_id":         run.get("drawing_id"),
            "spec_document_id":   run.get("spec_document_id"),
            "generated_at":       datetime.now(timezone.utc).isoformat(),
            "overall_grade":      overall_grade,
            "pass_rate":          round(pass_rate, 4),
            "summary":            summary,
            "counts":             counts,
            "severity_fails":     severity_fails,
            "spec_results":       spec_results,
            "ragas_summary":      ragas_summary,
            "recommendations":    recommendations,
        }

    # ── Data loaders ─────────────────────────────────────────── #

    def _load_run(self, spellcheck_run_id: int) -> Optional[dict]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, drawing_id, spec_document_id,
                           run_status, total_specs, specs_completed,
                           started_at, completed_at, model_family
                    FROM spellcheck_runs
                    WHERE id = %s
                    """,
                    (spellcheck_run_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def _load_spec_results(self, spellcheck_run_id: int) -> list[dict]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        ser.id,
                        ser.spec_rule_id,
                        ser.execution_mode,
                        ser.pass_fail,
                        ser.severity,
                        ser.confidence,
                        ser.issue_count,
                        ser.issue_summary,
                        ser.latency_ms,
                        sr.spec_code,
                        sr.spec_title,
                        sr.rule_type,
                        sr.normalized_rule_text
                    FROM spec_execution_runs ser
                    JOIN spec_rules sr ON sr.id = ser.spec_rule_id
                    WHERE ser.spellcheck_run_id = %s
                    ORDER BY
                        CASE ser.severity
                            WHEN 'critical' THEN 1
                            WHEN 'high'     THEN 2
                            WHEN 'medium'   THEN 3
                            WHEN 'low'      THEN 4
                            ELSE 5
                        END,
                        ser.pass_fail
                    """,
                    (spellcheck_run_id,),
                )
                return [dict(r) for r in cur.fetchall()]

    def _load_ragas_scores(self, spellcheck_run_id: int) -> list[dict]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        ses.metric_name,
                        AVG(ses.metric_value) AS avg_value,
                        COUNT(*) AS n
                    FROM spec_eval_scores ses
                    JOIN spec_execution_runs ser
                        ON ser.id = ses.spec_execution_run_id
                    WHERE ser.spellcheck_run_id = %s
                    GROUP BY ses.metric_name
                    ORDER BY ses.metric_name
                    """,
                    (spellcheck_run_id,),
                )
                return [dict(r) for r in cur.fetchall()]

    # ── Grade + summary ───────────────────────────────────────── #

    def _compute_grade(
        self,
        pass_rate:     float,
        severity_fails: dict,
        counts:        dict,
    ) -> str:
        """
        Compute overall grade. Critical failures automatically cap at D.
        High-severity failures cap at C.
        """
        if severity_fails.get("critical", 0) > 0:
            return "D" if pass_rate >= 0.50 else "F"
        if severity_fails.get("high", 0) > 2:
            base = self._grade_from_rate(pass_rate)
            return min(base, "C", key=lambda g: "ABCDF".index(g))

        return self._grade_from_rate(pass_rate)

    def _grade_from_rate(self, pass_rate: float) -> str:
        for grade, threshold in GRADE_THRESHOLDS.items():
            if pass_rate >= threshold:
                return grade
        return "F"

    def _build_summary(
        self,
        run:          dict,
        counts:       dict,
        total:        int,
        pass_rate:    float,
        grade:        str,
    ) -> str:
        pct = round(pass_rate * 100, 1)
        parts = [
            f"Drawing checked against {total} specification rules.",
            f"Overall grade: {grade} ({pct}% pass rate).",
            f"{counts['pass']} passed, {counts['fail']} failed, "
            f"{counts['warning']} warnings, {counts['needs_review']} need review.",
        ]
        if counts["fail"] == 0 and counts["warning"] == 0:
            parts.append("No compliance issues detected.")
        elif counts.get("fail", 0) > 0:
            parts.append(
                f"{counts['fail']} specification(s) require attention before approval."
            )
        return " ".join(parts)

    def _summarise_ragas(self, ragas_scores: list[dict]) -> dict:
        """Convert raw RAGAS rows into a summary dict."""
        summary = {}
        for row in ragas_scores:
            metric = row.get("metric_name", "")
            value  = row.get("avg_value")
            if value is not None:
                summary[metric] = round(float(value), 4)
        return summary

    def _build_recommendations(
        self,
        counts:         dict,
        severity_fails: dict,
        ragas_summary:  dict,
    ) -> list[str]:
        recs = []

        if severity_fails.get("critical", 0) > 0:
            recs.append(
                "CRITICAL: Address all critical-severity failures before this "
                "drawing can be approved for production use."
            )
        if severity_fails.get("high", 0) > 0:
            recs.append(
                f"{severity_fails['high']} high-severity issue(s) require "
                "engineering review."
            )
        if counts.get("needs_review", 0) > 0:
            recs.append(
                f"{counts['needs_review']} spec check(s) returned needs_review — "
                "verify retrieval recipes and evidence availability."
            )

        rr = ragas_summary.get("retrieval_relevance")
        if rr is not None and rr < 0.60:
            recs.append(
                "Low retrieval relevance scores detected. Consider refining "
                "retrieval_recipe configurations for affected rules."
            )

        fpr = ragas_summary.get("false_positive_risk")
        if fpr is not None and fpr > 0.50:
            recs.append(
                "High false positive risk detected. Review LLM-judged results "
                "for unsupported issue claims."
            )

        if not recs:
            recs.append(
                "Drawing meets all checked specifications. "
                "Proceed with standard approval workflow."
            )

        return recs