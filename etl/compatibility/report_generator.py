"""
cadsentinel.etl.compatibility.report_generator
------------------------------------------------
Generates a compatibility report from a CompatibilityResult.
Combines spellcheck results with dimensional compatibility findings.
"""

from __future__ import annotations

import logging
from datetime import datetime

from .compatibility_checker import CompatibilityResult, CompatibilityIssue
from .spec_extractor import DrawingSpecs

log = logging.getLogger(__name__)


def generate_compatibility_report(
    result:      CompatibilityResult,
    specs_list:  list[DrawingSpecs],
    run_results: dict[int, dict],
) -> dict:
    """
    Generate a full compatibility report combining:
    - Dimensional compatibility findings
    - Spellcheck compliance results per drawing

    Args:
        result:      CompatibilityResult from check_compatibility
        specs_list:  DrawingSpecs for each drawing
        run_results: dict of drawing_id -> spellcheck run result

    Returns:
        Report dict ready for JSON serialization
    """
    now = datetime.utcnow().isoformat() + "Z"

    # Build per-drawing summary
    drawing_summaries = []
    for specs in specs_list:
        run = run_results.get(specs.drawing_id, {})
        total = run.get("total_specs", 0)
        passed = run.get("pass_count", 0)
        pct = round(passed / total * 100) if total > 0 else 0

        # Get compatibility issues for this drawing
        drawing_issues = [
            {
                "issue_type":  i.issue_type,
                "description": i.description,
                "severity":    i.severity,
                "expected":    i.expected,
                "found":       i.found,
            }
            for i in result.issues
            if i.drawing_id == specs.drawing_id
        ]

        drawing_summaries.append({
            "drawing_id":         specs.drawing_id,
            "filename":           specs.filename,
            "drawing_type":       specs.drawing_type,
            "bore":               specs.bore,
            "stroke":             specs.stroke,
            "rod":                specs.rod,
            "ports":              specs.ports,
            "model_code":         specs.model_code,
            "spellcheck_run_id":  run.get("spellcheck_run_id"),
            "rules_checked":      total,
            "rules_passed":       passed,
            "rules_failed":       run.get("fail_count", 0),
            "rules_review":       run.get("review_count", 0),
            "pass_rate":          pct,
            "compatibility_issues": drawing_issues,
        })

    # Tally issues by severity
    critical = [i for i in result.issues if i.severity == "critical"]
    warnings = [i for i in result.issues if i.severity == "warning"]
    info     = [i for i in result.issues if i.severity == "info"]

    # Overall grade
    if critical:
        grade = "FAIL"
    elif warnings:
        grade = "REVIEW"
    else:
        grade = "PASS"

    return {
        "report_type":        "compatibility",
        "generated_at":       now,
        "folder_path":        result.folder_path,
        "grade":              grade,
        "summary":            result.summary,
        "drawings_checked":   result.drawings_checked,
        "assembly_count":     result.assembly_count,
        "issue_counts": {
            "critical": len(critical),
            "warning":  len(warnings),
            "info":     len(info),
            "total":    len(result.issues),
        },
        "compatibility_issues": [
            {
                "drawing_id":  i.drawing_id,
                "filename":    i.filename,
                "issue_type":  i.issue_type,
                "description": i.description,
                "severity":    i.severity,
                "expected":    i.expected,
                "found":       i.found,
            }
            for i in result.issues
        ],
        "drawings": drawing_summaries,
    }