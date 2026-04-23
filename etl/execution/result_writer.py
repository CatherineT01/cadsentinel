# result_writer.py

"""
cadsentinel.etl.execution.result_writer
-----------------------------------------
Writes spec execution results to the database.
Handles spec_execution_runs and spec_execution_issues tables.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

log = logging.getLogger(__name__)


def write_execution_result(
    cur,
    spellcheck_run_id: int,
    drawing_id:        int,
    spec_rule_id:      int,
    rule_version:      int,
    execution_mode:    str,
    result:            dict,
    evidence_package:  dict,
) -> Optional[int]:
    """
    Insert one spec_execution_runs row and its associated issues.

    Args:
        cur:               psycopg2 cursor (inside active transaction)
        spellcheck_run_id: parent run ID
        drawing_id:        drawing being checked
        spec_rule_id:      rule that was checked
        rule_version:      rule version number
        execution_mode:    deterministic | hybrid | llm_judge
        result:            dict from validator or llm_checker
        evidence_package:  full evidence package from retriever

    Returns:
        spec_execution_run_id (int) or None on failure
    """
    try:
        cur.execute(
            """
            INSERT INTO spec_execution_runs (
                spellcheck_run_id,
                drawing_id,
                spec_rule_id,
                rule_version,
                execution_mode,
                execution_status,
                pass_fail,
                severity,
                confidence,
                issue_count,
                issue_summary,
                detailed_result,
                retrieved_evidence,
                llm_raw_response,
                model_name,
                token_input,
                token_output,
                latency_ms,
                started_at,
                completed_at
            ) VALUES (
                %s,%s,%s,%s,%s,
                'complete',
                %s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,
                NOW(), NOW()
            )
            RETURNING id
            """,
            (
                spellcheck_run_id,
                drawing_id,
                spec_rule_id,
                rule_version,
                execution_mode,
                result.get("pass_fail", "needs_review"),
                result.get("severity", "medium"),
                result.get("confidence", 0.0),
                len(result.get("issues", [])),
                result.get("issue_summary", ""),
                json.dumps(result),
                json.dumps(evidence_package.get("evidence", []), default=str),
                json.dumps(result.get("llm_raw_response")) if result.get("llm_raw_response") else None,
                result.get("model_name"),
                result.get("token_input"),
                result.get("token_output"),
                result.get("latency_ms"),
            ),
        )
        run_id: int = cur.fetchone()["id"]

    except Exception:
        log.exception(
            "Failed to write spec_execution_run for spec_rule_id=%d", spec_rule_id
        )
        return None

    # Write issues
    issues = result.get("issues", [])
    for issue in issues:
        _write_issue(cur, run_id, issue)

    return run_id


def _write_issue(cur, spec_execution_run_id: int, issue: dict) -> None:
    """Insert one row into spec_execution_issues."""
    try:
        cur.execute(
            """
            INSERT INTO spec_execution_issues (
                spec_execution_run_id,
                issue_type,
                severity,
                description,
                suggested_fix,
                entity_ref,
                confidence
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                spec_execution_run_id,
                issue.get("issue_type", "unknown"),
                issue.get("severity", "medium"),
                issue.get("description", ""),
                issue.get("suggested_fix", ""),
                json.dumps(issue.get("entity_ref")) if issue.get("entity_ref") else None,
                issue.get("confidence", 1.0),
            ),
        )
    except Exception:
        log.exception(
            "Failed to write issue for spec_execution_run_id=%d", spec_execution_run_id
        )


def create_spellcheck_run(
    cur,
    drawing_id:       int,
    spec_document_id: int,
    total_specs:      int,
    triggered_by:     str = "system",
    model_family:     str = "",
) -> int:
    """
    Create a spellcheck_runs row and return its ID.
    Called once before the per-spec execution loop begins.
    """
    cur.execute(
        """
        INSERT INTO spellcheck_runs (
            drawing_id,
            spec_document_id,
            run_status,
            total_specs,
            specs_completed,
            triggered_by,
            model_family,
            started_at
        ) VALUES (%s, %s, 'running', %s, 0, %s, %s, NOW())
        RETURNING id
        """,
        (drawing_id, spec_document_id, total_specs, triggered_by, model_family),
    )
    return cur.fetchone()["id"]


def mark_run_complete(cur, spellcheck_run_id: int, specs_completed: int) -> None:
    """Update spellcheck_runs status to completed."""
    cur.execute(
        """
        UPDATE spellcheck_runs
        SET run_status      = 'completed',
            specs_completed = %s,
            completed_at    = NOW()
        WHERE id = %s
        """,
        (specs_completed, spellcheck_run_id),
    )


def mark_run_failed(cur, spellcheck_run_id: int, reason: str = "") -> None:
    """Update spellcheck_runs status to failed."""
    cur.execute(
        """
        UPDATE spellcheck_runs
        SET run_status   = 'failed',
            notes        = %s,
            completed_at = NOW()
        WHERE id = %s
        """,
        (reason[:500] if reason else "", spellcheck_run_id),
    )


def write_run_summary(cur, spellcheck_run_id: int) -> None:
    """
    Compute and insert a spellcheck_run_summaries row
    by aggregating all spec_execution_runs for this run.
    """
    try:
        cur.execute(
            """
            INSERT INTO spellcheck_run_summaries (
                spellcheck_run_id,
                pass_count,
                fail_count,
                warning_count,
                review_count,
                avg_confidence
            )
            SELECT
                %s,
                COUNT(*) FILTER (WHERE pass_fail = 'pass'),
                COUNT(*) FILTER (WHERE pass_fail = 'fail'),
                COUNT(*) FILTER (WHERE pass_fail = 'warning'),
                COUNT(*) FILTER (WHERE pass_fail = 'needs_review'),
                AVG(confidence)
            FROM spec_execution_runs
            WHERE spellcheck_run_id = %s
            ON CONFLICT (spellcheck_run_id) DO UPDATE SET
                pass_count    = EXCLUDED.pass_count,
                fail_count    = EXCLUDED.fail_count,
                warning_count = EXCLUDED.warning_count,
                review_count  = EXCLUDED.review_count,
                avg_confidence = EXCLUDED.avg_confidence
            """,
            (spellcheck_run_id, spellcheck_run_id),
        )
    except Exception:
        log.exception(
            "Failed to write run summary for spellcheck_run_id=%d",
            spellcheck_run_id
        )