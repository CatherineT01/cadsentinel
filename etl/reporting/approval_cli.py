# approval_cli.py

"""
cadsentinel.etl.reporting.approval_cli
----------------------------------------
Command-line tool for human approval of extracted spec rules.

Rules are stored with approved=False after LLM extraction.
A human reviewer must approve them before they are used in spellcheck runs.

Usage:
    python -m cadsentinel.etl.reporting.approval_cli list   --doc-id 3
    python -m cadsentinel.etl.reporting.approval_cli review --doc-id 3
    python -m cadsentinel.etl.reporting.approval_cli approve --rule-id 142
    python -m cadsentinel.etl.reporting.approval_cli approve-all --doc-id 3
    python -m cadsentinel.etl.reporting.approval_cli reject --rule-id 142
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from ..db import get_connection


def cmd_list(doc_id: int, approved: bool = False) -> None:
    """List spec rules for a document."""
    status_filter = "TRUE" if approved else "FALSE"
    label         = "approved" if approved else "pending approval"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, spec_code, spec_title, rule_type,
                       execution_mode, severity_default, approved
                FROM spec_rules
                WHERE spec_document_id = %s
                  AND approved = {status_filter}
                ORDER BY id
                """,
                (doc_id,),
            )
            rows = cur.fetchall()

    if not rows:
        print(f"No {label} rules found for document {doc_id}.")
        return

    print(f"\n{'─'*70}")
    print(f"  {label.upper()} RULES — spec_document_id={doc_id} ({len(rows)} total)")
    print(f"{'─'*70}")
    for row in rows:
        code  = row["spec_code"] or "—"
        title = (row["spec_title"] or "untitled")[:45]
        print(
            f"  [{row['id']:>5}]  {code:<8}  {title:<45}  "
            f"{row['rule_type']:<18}  {row['execution_mode']:<14}  "
            f"{row['severity_default']}"
        )
    print(f"{'─'*70}\n")


def cmd_review(doc_id: int) -> None:
    """Interactive review — show each rule and prompt for approval."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, spec_code, spec_title, original_spec_text,
                       normalized_rule_text, rule_type, execution_mode,
                       severity_default, extraction_confidence
                FROM spec_rules
                WHERE spec_document_id = %s AND approved = FALSE
                ORDER BY id
                """,
                (doc_id,),
            )
            rows = cur.fetchall()

    if not rows:
        print(f"No pending rules for document {doc_id}. All caught up!")
        return

    approved_ids = []
    rejected_ids = []

    for i, row in enumerate(rows, 1):
        print(f"\n{'='*70}")
        print(f"  Rule {i}/{len(rows)}  —  ID: {row['id']}")
        print(f"{'='*70}")
        print(f"  Code:       {row['spec_code'] or '—'}")
        print(f"  Title:      {row['spec_title'] or '—'}")
        print(f"  Type:       {row['rule_type']}")
        print(f"  Mode:       {row['execution_mode']}")
        print(f"  Severity:   {row['severity_default']}")
        print(f"  Confidence: {row['extraction_confidence']}")
        print(f"\n  Original text:")
        print(f"  {row['original_spec_text'][:300]}")
        print(f"\n  Normalized:")
        print(f"  {row['normalized_rule_text'][:300]}")
        print()

        while True:
            choice = input("  [a]pprove / [r]eject / [s]kip / [q]uit: ").strip().lower()
            if choice in ("a", "r", "s", "q"):
                break
            print("  Please enter a, r, s, or q.")

        if choice == "a":
            approved_ids.append(row["id"])
            print("  → Approved")
        elif choice == "r":
            rejected_ids.append(row["id"])
            print("  → Rejected (will be deleted)")
        elif choice == "s":
            print("  → Skipped")
        elif choice == "q":
            print("\n  Quitting review.")
            break

    # Apply decisions
    if approved_ids or rejected_ids:
        _apply_decisions(approved_ids, rejected_ids)

    print(f"\nReview complete: {len(approved_ids)} approved, "
          f"{len(rejected_ids)} rejected, "
          f"{len(rows) - len(approved_ids) - len(rejected_ids)} skipped.\n")


def cmd_approve(rule_id: int, reviewer: str = "human") -> None:
    """Approve a single spec rule by ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE spec_rules
                SET approved    = TRUE,
                    approved_by = %s,
                    approved_at = %s
                WHERE id = %s
                RETURNING id, spec_title
                """,
                (reviewer, datetime.now(timezone.utc), rule_id),
            )
            row = cur.fetchone()
            conn.commit()

    if row:
        print(f"Approved rule {rule_id}: {row['spec_title'] or '—'}")
    else:
        print(f"Rule {rule_id} not found.")


def cmd_approve_all(doc_id: int, reviewer: str = "human") -> None:
    """Approve all pending rules for a spec document."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE spec_rules
                SET approved    = TRUE,
                    approved_by = %s,
                    approved_at = %s
                WHERE spec_document_id = %s AND approved = FALSE
                """,
                (reviewer, datetime.now(timezone.utc), doc_id),
            )
            count = cur.rowcount
            conn.commit()

    print(f"Approved {count} rules for spec_document_id={doc_id}.")


def cmd_reject(rule_id: int) -> None:
    """Delete a spec rule by ID (rejection removes it from the system)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM spec_rules WHERE id = %s RETURNING id, spec_title",
                (rule_id,),
            )
            row = cur.fetchone()
            conn.commit()

    if row:
        print(f"Rejected and deleted rule {rule_id}: {row['spec_title'] or '—'}")
    else:
        print(f"Rule {rule_id} not found.")


def _apply_decisions(approved_ids: list[int], rejected_ids: list[int]) -> None:
    now = datetime.now(timezone.utc)
    with get_connection() as conn:
        with conn.cursor() as cur:
            if approved_ids:
                cur.execute(
                    """
                    UPDATE spec_rules
                    SET approved = TRUE, approved_by = 'human', approved_at = %s
                    WHERE id = ANY(%s)
                    """,
                    (now, approved_ids),
                )
            if rejected_ids:
                cur.execute(
                    "DELETE FROM spec_rules WHERE id = ANY(%s)",
                    (rejected_ids,),
                )
            conn.commit()


# ── CLI entry point ───────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CADSentinel spec rule approval tool"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List rules for a document")
    p_list.add_argument("--doc-id", type=int, required=True)
    p_list.add_argument("--approved", action="store_true",
                        help="Show approved rules instead of pending")

    # review
    p_review = subparsers.add_parser("review", help="Interactive review")
    p_review.add_argument("--doc-id", type=int, required=True)

    # approve
    p_approve = subparsers.add_parser("approve", help="Approve a rule by ID")
    p_approve.add_argument("--rule-id", type=int, required=True)
    p_approve.add_argument("--reviewer", default="human")

    # approve-all
    p_all = subparsers.add_parser("approve-all", help="Approve all pending rules")
    p_all.add_argument("--doc-id", type=int, required=True)
    p_all.add_argument("--reviewer", default="human")

    # reject
    p_reject = subparsers.add_parser("reject", help="Reject and delete a rule")
    p_reject.add_argument("--rule-id", type=int, required=True)

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args.doc_id, args.approved)
    elif args.command == "review":
        cmd_review(args.doc_id)
    elif args.command == "approve":
        cmd_approve(args.rule_id, args.reviewer)
    elif args.command == "approve-all":
        cmd_approve_all(args.doc_id, args.reviewer)
    elif args.command == "reject":
        cmd_reject(args.rule_id)


if __name__ == "__main__":
    main()