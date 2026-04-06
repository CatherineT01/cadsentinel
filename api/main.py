"""
cadsentinel.api.main
---------------------
FastAPI service exposing all CADSentinel endpoints.

Install FastAPI: pip install fastapi uvicorn python-multipart

Run:
    uvicorn cadsentinel.api.main:app --reload --port 8000

Endpoints:
    POST /api/v1/specs/upload
    POST /api/v1/specs/{document_id}/extract
    GET  /api/v1/specs/{document_id}
    GET  /api/v1/spec-rules/{spec_rule_id}
    POST /api/v1/spec-rules/{spec_rule_id}/approve
    GET  /api/v1/spec-rules/pending/{document_id}

    POST /api/v1/spellcheck/run
    GET  /api/v1/spellcheck/runs/{run_id}
    GET  /api/v1/spellcheck/runs/{run_id}/results
    GET  /api/v1/spellcheck/runs/{run_id}/report

    POST /api/v1/evals/run/{spellcheck_run_id}
    GET  /api/v1/evals/runs/{spellcheck_run_id}

    GET  /api/v1/health
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    class FastAPI:  # type: ignore
        def get(self, *a, **kw): return lambda f: f
        def post(self, *a, **kw): return lambda f: f
    try:
        from pydantic import BaseModel
    except ImportError:
        class BaseModel:  # type: ignore
            pass
    HTTPException = Exception  # type: ignore

from ..etl.db import get_connection
from ..etl.spec_ingestor import SpecIngestor
from ..etl.execution.engine import SpellcheckEngine
from ..etl.evaluation.eval_writer import evaluate_spellcheck_run
from ..etl.reporting.report_generator import ReportGenerator

app = FastAPI(
    title       = "CADSentinel",
    description = "Automated CAD drawing compliance checking system",
    version     = "1.0.0",
)

LLM_PROVIDER = os.environ.get("CADSENTINEL_LLM_PROVIDER", "openai")


# ── Request models ────────────────────────────────────────────── #

class SpellcheckRunRequest(BaseModel):
    drawing_id:       int
    spec_document_id: int
    triggered_by:     str = "api"

class ApproveRuleRequest(BaseModel):
    approved_by: str = "human"

class EvalRunRequest(BaseModel):
    use_embeddings: bool = True
    gold_labels:    Optional[dict] = None


# ── Health ────────────────────────────────────────────────────── #

@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "cadsentinel"}


# ── Specification endpoints ───────────────────────────────────── #

@app.post("/api/v1/specs/upload")
async def upload_spec(
    background_tasks: BackgroundTasks,
    file:             UploadFile = File(...),
    skip_extraction:  bool       = Form(False),
    provider:         str        = Form(LLM_PROVIDER),
):
    """Upload a specification document and trigger extraction."""
    suffix = Path(file.filename or "doc.pdf").suffix.lower()
    if suffix not in (".pdf", ".docx", ".txt"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Use .pdf, .docx, or .txt"
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        ingestor = SpecIngestor(provider=provider, skip_extraction=skip_extraction)
        result   = ingestor.ingest(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


@app.get("/api/v1/specs/{document_id}")
def get_spec_document(document_id: int):
    """Get spec document metadata and rule counts."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, filename, source_type, title,
                       parse_status, uploaded_at
                FROM spec_documents WHERE id = %s
                """,
                (document_id,),
            )
            doc = cur.fetchone()
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found")

            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE approved = TRUE)  AS approved_count,
                    COUNT(*) FILTER (WHERE approved = FALSE) AS pending_count
                FROM spec_rules WHERE spec_document_id = %s
                """,
                (document_id,),
            )
            counts = cur.fetchone()

    return {
        **dict(doc),
        "approved_rules": counts["approved_count"],
        "pending_rules":  counts["pending_count"],
    }


@app.get("/api/v1/spec-rules/pending/{document_id}")
def list_pending_rules(document_id: int):
    """List all unapproved spec rules for a document."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, spec_code, spec_title, rule_type,
                       execution_mode, severity_default,
                       extraction_confidence, created_at
                FROM spec_rules
                WHERE spec_document_id = %s AND approved = FALSE
                ORDER BY id
                """,
                (document_id,),
            )
            rows = cur.fetchall()
    return {"document_id": document_id, "rules": [dict(r) for r in rows]}


@app.get("/api/v1/spec-rules/{spec_rule_id}")
def get_spec_rule(spec_rule_id: int):
    """Get full details of a spec rule."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM spec_rules WHERE id = %s",
                (spec_rule_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Rule not found")
    return dict(row)


@app.post("/api/v1/spec-rules/{spec_rule_id}/approve")
def approve_spec_rule(spec_rule_id: int, body: ApproveRuleRequest):
    """Approve a spec rule for use in spellcheck runs."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE spec_rules
                SET approved    = TRUE,
                    approved_by = %s,
                    approved_at = NOW()
                WHERE id = %s
                RETURNING id, spec_title
                """,
                (body.approved_by, spec_rule_id),
            )
            row = cur.fetchone()
            conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Rule not found")

    return {"approved": True, "rule_id": row["id"], "title": row["spec_title"]}


# ── Spellcheck endpoints ──────────────────────────────────────── #

@app.post("/api/v1/spellcheck/run")
def start_spellcheck_run(body: SpellcheckRunRequest):
    """Start a spellcheck run for a drawing against a spec document."""
    engine = SpellcheckEngine(provider=LLM_PROVIDER)
    try:
        result = engine.run(
            drawing_id       = body.drawing_id,
            spec_document_id = body.spec_document_id,
            triggered_by     = body.triggered_by,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@app.get("/api/v1/spellcheck/runs/{run_id}")
def get_spellcheck_run(run_id: int):
    """Get status and summary of a spellcheck run."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sr.*, srs.pass_count, srs.fail_count,
                       srs.warning_count, srs.review_count,
                       srs.avg_confidence, srs.final_grade
                FROM spellcheck_runs sr
                LEFT JOIN spellcheck_run_summaries srs ON srs.spellcheck_run_id = sr.id
                WHERE sr.id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Run not found")
    return dict(row)


@app.get("/api/v1/spellcheck/runs/{run_id}/results")
def get_spellcheck_results(run_id: int):
    """Get all per-spec results for a spellcheck run."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    ser.id, ser.spec_rule_id, ser.execution_mode,
                    ser.pass_fail, ser.severity, ser.confidence,
                    ser.issue_count, ser.issue_summary,
                    sr.spec_code, sr.spec_title, sr.rule_type
                FROM spec_execution_runs ser
                JOIN spec_rules sr ON sr.id = ser.spec_rule_id
                WHERE ser.spellcheck_run_id = %s
                ORDER BY ser.id
                """,
                (run_id,),
            )
            rows = cur.fetchall()
    return {"run_id": run_id, "results": [dict(r) for r in rows]}


@app.get("/api/v1/spellcheck/runs/{run_id}/report")
def get_compliance_report(run_id: int):
    """Generate and return a full compliance report."""
    generator = ReportGenerator()
    try:
        report = generator.generate(run_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])

    return report


# ── Evaluation endpoints ──────────────────────────────────────── #

@app.post("/api/v1/evals/run/{spellcheck_run_id}")
def run_evaluation(spellcheck_run_id: int, body: EvalRunRequest):
    """Trigger RAGAS evaluation for a completed spellcheck run."""
    try:
        result = evaluate_spellcheck_run(
            spellcheck_run_id = spellcheck_run_id,
            use_embeddings    = body.use_embeddings,
            gold_labels       = body.gold_labels,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@app.get("/api/v1/evals/runs/{spellcheck_run_id}")
def get_evaluation_scores(spellcheck_run_id: int):
    """Get aggregated RAGAS scores for a spellcheck run."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    ses.metric_name,
                    AVG(ses.metric_value) AS avg_value,
                    MIN(ses.metric_value) AS min_value,
                    MAX(ses.metric_value) AS max_value,
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
            rows = cur.fetchall()

    return {
        "spellcheck_run_id": spellcheck_run_id,
        "scores": [dict(r) for r in rows],
    }