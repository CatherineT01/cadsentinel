"""
cadsentinel.etl.compatibility.pipeline
----------------------------------------
Main entry point for the folder compatibility checker.
"""

from __future__ import annotations
import logging
log = logging.getLogger(__name__)


def run_folder_check(
    folder_path:      str,
    inspector_path:   str,
    spec_document_id: int  = 1,
    provider:         str  = "openai",
    force_reingest:   bool = False,
) -> dict:
    """Run a full folder compatibility check."""
    from .folder_scanner import scan_folder
    from .spec_extractor import extract_specs_batch
    from .compatibility_checker import check_compatibility
    from .report_generator import generate_compatibility_report
    from ..execution.engine import SpellcheckEngine

    log.info(f"Starting folder check: {folder_path}")

    # Step 1 — scan and ingest
    log.info("Step 1: Scanning folder...")
    scan_results = scan_folder(
        folder_path    = folder_path,
        inspector_path = inspector_path,
        provider       = provider,
        force_reingest = force_reingest,
    )

    successful = [r for r in scan_results if r.drawing_id is not None]
    failed     = [r for r in scan_results if r.drawing_id is None]

    if failed:
        log.warning(f"{len(failed)} file(s) failed to ingest: {[r.filename for r in failed]}")

    if not successful:
        return {"error": "No drawings could be ingested.", "folder_path": folder_path}

    drawing_ids = [r.drawing_id for r in successful]
    log.info(f"Step 1 complete: {len(successful)} drawings ready")

    # Step 2 — spellcheck each drawing
    log.info("Step 2: Running spellcheck on each drawing...")
    engine      = SpellcheckEngine(provider=provider, embed_for_retrieval=False)
    run_results = {}

    for drawing_id in drawing_ids:
        try:
            result = engine.run(
                drawing_id       = drawing_id,
                spec_document_id = spec_document_id,
                triggered_by     = "folder_check",
            )
            run_results[drawing_id] = result
            log.info(f"  id={drawing_id} pass={result['pass_count']} fail={result['fail_count']} review={result['review_count']}")
        except Exception as e:
            log.warning(f"Spellcheck failed for drawing {drawing_id}: {e}")

    log.info(f"Step 2 complete: {len(run_results)} spellchecks run")

    # Step 3 — extract dimensional specs
    log.info("Step 3: Extracting dimensional specs...")
    specs_list = extract_specs_batch(drawing_ids)
    log.info(f"Step 3 complete: {len(specs_list)} spec sets extracted")

    # Step 4 — check compatibility
    log.info("Step 4: Checking dimensional compatibility...")
    compat_result = check_compatibility(specs_list, folder_path=folder_path)
    log.info(f"Step 4 complete: {len(compat_result.issues)} issue(s) found")

    # Step 5 — generate report
    log.info("Step 5: Generating report...")
    report = generate_compatibility_report(
        result      = compat_result,
        specs_list  = specs_list,
        run_results = run_results,
    )

    log.info(f"Folder check complete: grade={report['grade']}")
    return report
