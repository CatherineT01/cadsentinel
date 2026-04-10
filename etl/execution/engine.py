# engine.py

"""
cadsentinel.etl.execution.engine
----------------------------------
Spec execution engine.

Orchestrates a full spellcheck run:
  1. Load approved spec rules for a document
  2. Create a spellcheck_runs row
  3. For each spec rule (in parallel):
     a. Retrieve targeted evidence
     b. Route to correct executor (deterministic / hybrid / llm_judge)
     c. Store result + issues
  4. Write run summary
  5. Mark run complete

Usage:
    engine = SpellcheckEngine(provider="openai")
    result = engine.run(drawing_id=7, spec_document_id=3)
    print(result["pass_count"], result["fail_count"])
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from ..db import get_connection
from ..retriever import EvidenceRetriever
from ..validators import (
    TitleBlockValidator,
    ConfidentialityValidator,
    LayerNamingValidator,
    DimensionUnitsValidator,
    RevisionTableValidator,
    ModelCodeValidator,
    StandardNotesValidator,
    CylinderSpecValidator,
    JITBoreValidator,
    JITMountValidator,
)
from .llm_checker import llm_check
from .result_writer import (
    create_spellcheck_run,
    write_execution_result,
    mark_run_complete,
    mark_run_failed,
    write_run_summary,
)

log = logging.getLogger(__name__)

MAX_WORKERS    = int(os.environ.get("CADSENTINEL_MAX_WORKERS", "4"))
LLM_PROVIDER   = os.environ.get("CADSENTINEL_LLM_PROVIDER", "openai").lower()

# Map rule_type → validator class
DETERMINISTIC_VALIDATORS = {
    "title_block":      TitleBlockValidator(),
    "layer_naming":     LayerNamingValidator(),
    "dimension_units":  DimensionUnitsValidator(),
    "revision_table":   RevisionTableValidator(),
    "block_naming":     LayerNamingValidator(),
    "text_style":       TitleBlockValidator(),
    "model_code":       ModelCodeValidator(),
    "standard_notes":   StandardNotesValidator(),
    "cylinder_spec":    CylinderSpecValidator(),
    "jit_bore":         JITBoreValidator(),
    "jit_mount":        JITMountValidator(),
    "confidentiality": ConfidentialityValidator(),
}


class SpellcheckEngine:
    """
    Orchestrates a full spellcheck run for one drawing against one spec document.

    Args:
        provider:            "openai" or "grok" for LLM checks
        embed_for_retrieval: Whether to use vector search in evidence retrieval
        max_workers:         Thread pool size for parallel spec execution
    """

    def __init__(
        self,
        provider:            str  = LLM_PROVIDER,
        embed_for_retrieval: bool = True,
        max_workers:         int  = MAX_WORKERS,
    ):
        self.provider            = provider
        self.max_workers         = max_workers
        self.retriever           = EvidenceRetriever(
            embed_for_vector_search=embed_for_retrieval
        )

    # ── Public API ──────────────────────────────────────────────── #

    def run(
        self,
        drawing_id:       int,
        spec_document_id: int,
        triggered_by:     str = "system",
    ) -> dict:
        """
        Execute all approved spec rules for a drawing.

        Returns:
            dict with spellcheck_run_id, pass_count, fail_count,
                  warning_count, review_count, total_specs
        """
        log.info(
            "Starting spellcheck run: drawing_id=%d spec_document_id=%d",
            drawing_id, spec_document_id,
        )

        # Load approved rules
        # Classify drawing type and filter rules accordingly
        from ..classifiers.type_store import get_drawing_type
        stored_type = get_drawing_type(drawing_id)
        drawing_type_code = stored_type.type_code if stored_type else None
        log.info(f"Drawing {drawing_id} type: {drawing_type_code}")
        rules = self._load_approved_rules(spec_document_id, drawing_type_code)
        if not rules:
            log.warning(
                "No approved rules found for spec_document_id=%d",
                spec_document_id,
            )
            return {
                "spellcheck_run_id": None,
                "total_specs":       0,
                "pass_count":        0,
                "fail_count":        0,
                "warning_count":     0,
                "review_count":      0,
                "error":             "No approved rules found",
            }

        # Create spellcheck_runs row
        with get_connection() as conn:
            with conn.cursor() as cur:
                spellcheck_run_id = create_spellcheck_run(
                    cur,
                    drawing_id       = drawing_id,
                    spec_document_id = spec_document_id,
                    total_specs      = len(rules),
                    triggered_by     = triggered_by,
                    model_family     = self.provider,
                )
                conn.commit()

        log.info(
            "Created spellcheck_run_id=%d for %d rules",
            spellcheck_run_id, len(rules),
        )

        # Execute all specs in parallel
        results = self._execute_parallel(
            rules             = rules,
            drawing_id        = drawing_id,
            spellcheck_run_id = spellcheck_run_id,
        )

        # Tally results
        counts = {"pass": 0, "fail": 0, "warning": 0, "needs_review": 0}
        for r in results:
            verdict = r.get("pass_fail", "needs_review")
            counts[verdict] = counts.get(verdict, 0) + 1

        specs_completed = len(results)

        # Write summary and mark complete
        with get_connection() as conn:
            with conn.cursor() as cur:
                write_run_summary(cur, spellcheck_run_id)
                mark_run_complete(cur, spellcheck_run_id, specs_completed)
                conn.commit()

        log.info(
            "Spellcheck run %d complete: pass=%d fail=%d warning=%d review=%d",
            spellcheck_run_id,
            counts["pass"], counts["fail"],
            counts["warning"], counts["needs_review"],
        )

        return {
            "spellcheck_run_id": spellcheck_run_id,
            "total_specs":       len(rules),
            "specs_completed":   specs_completed,
            "pass_count":        counts["pass"],
            "fail_count":        counts["fail"],
            "warning_count":     counts["warning"],
            "review_count":      counts["needs_review"],
        }

    # ── Parallel execution ───────────────────────────────────────── #

    def _execute_parallel(
        self,
        rules:             list[dict],
        drawing_id:        int,
        spellcheck_run_id: int,
    ) -> list[dict]:
        """
        Execute all spec rules in parallel using a thread pool.
        Returns list of result dicts (one per rule).
        """
        results: list[dict] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_rule = {
                executor.submit(
                    self._execute_one_rule,
                    rule              = rule,
                    drawing_id        = drawing_id,
                    spellcheck_run_id = spellcheck_run_id,
                ): rule
                for rule in rules
            }

            for future in as_completed(future_to_rule):
                rule = future_to_rule[future]
                try:
                    result = future.result()
                    results.append(result)
                    log.debug(
                        "Rule %s → %s (confidence=%.2f)",
                        rule.get("spec_code", rule["id"]),
                        result.get("pass_fail"),
                        result.get("confidence", 0),
                    )
                except Exception:
                    log.exception(
                        "Unhandled error executing rule id=%d", rule["id"]
                    )
                    results.append({"pass_fail": "needs_review", "confidence": 0.0})

        return results

    # ── Single rule execution ────────────────────────────────────── #

    def _execute_one_rule(
        self,
        rule:              dict,
        drawing_id:        int,
        spellcheck_run_id: int,
    ) -> dict:
        """
        Execute one spec rule against one drawing.
        Retrieves evidence, routes to correct executor, stores result.
        """
        spec_rule_id    = rule["id"]
        execution_mode  = rule["execution_mode"]
        rule_type       = rule["rule_type"]
        severity        = rule["severity_default"]
        retrieval_recipe = rule.get("retrieval_recipe") or {}
        structured_rule  = rule.get("structured_rule") or {}
        rule_config      = {**structured_rule, "severity_default": severity}

        log.debug(
            "Executing rule id=%d type=%s mode=%s",
            spec_rule_id, rule_type, execution_mode,
        )

        # Step 1 — retrieve evidence
        try:
            evidence_package = self.retriever.retrieve(
                drawing_id        = drawing_id,
                spec_rule_id      = spec_rule_id,
                normalized_rule_text = rule["normalized_rule_text"],
                retrieval_recipe  = retrieval_recipe,
                checker_prompt    = rule.get("checker_prompt"),
            )
        except Exception:
            log.exception("Evidence retrieval failed for rule id=%d", spec_rule_id)
            evidence_package = {"evidence": [], "evidence_count": 0}

        # Step 2 — execute check
        result = self._route_and_execute(
            execution_mode   = execution_mode,
            rule_type        = rule_type,
            rule_config      = rule_config,
            rule             = rule,
            evidence_package = evidence_package,
        )

        # Step 3 — store result
        with get_connection() as conn:
            with conn.cursor() as cur:
                write_execution_result(
                    cur               = cur,
                    spellcheck_run_id = spellcheck_run_id,
                    drawing_id        = drawing_id,
                    spec_rule_id      = spec_rule_id,
                    rule_version      = rule.get("rule_version", 1),
                    execution_mode    = execution_mode,
                    result            = result,
                    evidence_package  = evidence_package,
                )
                conn.commit()

        return result

    def _route_and_execute(
        self,
        execution_mode:   str,
        rule_type:        str,
        rule_config:      dict,
        rule:             dict,
        evidence_package: dict,
    ) -> dict:
        """Route to the correct executor based on execution_mode."""

        if execution_mode == "deterministic":
            return self._run_deterministic(
                rule_type        = rule_type,
                rule_config      = rule_config,
                evidence_package = evidence_package,
            )

        elif execution_mode in ("hybrid", "llm_judge"):
            return llm_check(
                evidence_package = evidence_package,
                rule_text        = rule["normalized_rule_text"],
                rule_type        = rule_type,
                severity         = rule_config.get("severity_default", "medium"),
                checker_prompt   = rule.get("checker_prompt"),
                provider         = self.provider,
            )

        else:
            log.warning(
                "Unknown execution_mode '%s' for rule type '%s' — defaulting to hybrid",
                execution_mode, rule_type,
            )
            return llm_check(
                evidence_package = evidence_package,
                rule_text        = rule["normalized_rule_text"],
                rule_type        = rule_type,
                severity         = rule_config.get("severity_default", "medium"),
                checker_prompt   = rule.get("checker_prompt"),
                provider         = self.provider,
            )

    def _run_deterministic(
        self,
        rule_type:        str,
        rule_config:      dict,
        evidence_package: dict,
    ) -> dict:
        """Route to the correct deterministic validator."""
        validator = DETERMINISTIC_VALIDATORS.get(rule_type)

        if validator is None:
            log.warning(
                "No deterministic validator for rule_type='%s' — routing to hybrid",
                rule_type,
            )
            # Fall back to LLM if no validator exists for this rule type
            return {
                "pass_fail":     "needs_review",
                "confidence":    0.0,
                "severity":      rule_config.get("severity_default", "medium"),
                "issue_summary": f"No deterministic validator for rule_type '{rule_type}'",
                "issues":        [],
                "evidence_used": [],
            }

        result = validator.validate(evidence_package, rule_config)
        return result.to_dict()

    # ── DB helpers ───────────────────────────────────────────────── #

    def _load_approved_rules(
        self,
        spec_document_id: int,
        drawing_type_code: str | None = None,
    ) -> list[dict]:
        """Load approved spec rules, filtered by drawing type if provided."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                if drawing_type_code and drawing_type_code != "unknown":
                    cur.execute(
                        """
                        SELECT
                            sr.id,
                            sr.spec_code,
                            sr.spec_title,
                            sr.normalized_rule_text,
                            sr.rule_type,
                            sr.execution_mode,
                            sr.severity_default,
                            sr.entity_scope,
                            sr.expected_evidence_types,
                            sr.checker_prompt,
                            sr.retrieval_recipe,
                            sr.structured_rule,
                            sr.rule_version
                        FROM spec_rules sr
                        JOIN spec_rule_drawing_types srdt ON srdt.spec_rule_id = sr.id
                        JOIN drawing_types dt ON dt.id = srdt.drawing_type_id
                        WHERE sr.spec_document_id = %s
                          AND sr.approved = TRUE
                          AND dt.type_code = %s
                        ORDER BY sr.id
                        """,
                        (spec_document_id, drawing_type_code),
                    )
                else:
                    cur.execute(
                        """
                        SELECT
                            id,
                            spec_code,
                            spec_title,
                            normalized_rule_text,
                            rule_type,
                            execution_mode,
                            severity_default,
                            entity_scope,
                            expected_evidence_types,
                            checker_prompt,
                            retrieval_recipe,
                            structured_rule,
                            rule_version
                        FROM spec_rules
                        WHERE spec_document_id = %s
                          AND approved = TRUE
                        ORDER BY id
                        """,
                        (spec_document_id,),
                    )
                rows = cur.fetchall()
        return [dict(r) for r in rows]