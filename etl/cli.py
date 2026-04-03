"""
cadsentinel.etl.cli
-------------------
Command-line interface for the ETL ingestor.

Usage:
    python -m cadsentinel.etl.cli ingest drawing.dwg
    python -m cadsentinel.etl.cli ingest drawing.dwg --embed
    python -m cadsentinel.etl.cli ingest-dir /path/to/drawings/ --embed
    python -m cadsentinel.etl.cli embed
    python -m cadsentinel.etl.cli embed --drawing-id 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .ingestor import DwgIngestor, IngestionError
from .embedder import embed_pending

log = logging.getLogger(__name__)


def cmd_ingest(args):
    ingestor   = DwgIngestor(inspector_path=args.inspector)
    drawing_id = ingestor.ingest(args.dwg_file)
    print(f"Ingested: drawing_id={drawing_id}")
    if args.embed:
        print("Embedding text chunks...")
        count = embed_pending(drawing_id=drawing_id)
        print(f"Embedded {count} text chunks.")


def cmd_ingest_dir(args):
    ingestor  = DwgIngestor(inspector_path=args.inspector)
    directory = Path(args.directory)
    dwg_files = list(directory.glob("**/*.dwg")) + list(directory.glob("**/*.DWG"))

    if not dwg_files:
        print(f"No DWG files found in: {directory}")
        sys.exit(1)

    print(f"Found {len(dwg_files)} DWG files.")
    success, failed = 0, 0

    for dwg_path in dwg_files:
        try:
            drawing_id = ingestor.ingest(dwg_path)
            print(f"  OK  drawing_id={drawing_id}  {dwg_path.name}")
            success += 1
        except IngestionError as exc:
            print(f"  FAIL  {dwg_path.name}: {exc}")
            failed += 1

    print(f"\nComplete: {success} succeeded, {failed} failed.")
    if args.embed and success > 0:
        print("Embedding all pending text chunks...")
        count = embed_pending()
        print(f"Embedded {count} text chunks.")


def cmd_embed(args):
    count = embed_pending(drawing_id=args.drawing_id, dry_run=args.dry_run)
    print(f"Done. Embedded {count} chunks.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="CADSentinel ETL — DWG ingestion and embedding pipeline"
    )
    parser.add_argument(
        "--inspector",
        default="./dwg_inspect",
        help="Path to dwg_inspect binary (default: ./dwg_inspect)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_ingest = subparsers.add_parser("ingest", help="Ingest a single DWG file")
    p_ingest.add_argument("dwg_file", help="Path to the .dwg file")
    p_ingest.add_argument("--embed", action="store_true",
                          help="Embed text chunks immediately after ingestion")
    p_ingest.set_defaults(func=cmd_ingest)

    p_dir = subparsers.add_parser("ingest-dir", help="Ingest all DWG files in a directory")
    p_dir.add_argument("directory", help="Directory to scan for .dwg files")
    p_dir.add_argument("--embed", action="store_true",
                       help="Embed text chunks after all ingestions complete")
    p_dir.set_defaults(func=cmd_ingest_dir)

    p_embed = subparsers.add_parser("embed", help="Embed pending text chunks")
    p_embed.add_argument("--drawing-id", type=int, default=None,
                         help="Only embed chunks for this drawing ID")
    p_embed.add_argument("--dry-run", action="store_true",
                         help="Show what would be embedded without calling the API")
    p_embed.set_defaults(func=cmd_embed)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()