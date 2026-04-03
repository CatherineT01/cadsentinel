"""
cadsentinel.etl.embedder
------------------------
Embedding worker: finds all drawing_text_chunks rows where embedding IS NULL
and fills them using the configured embedding model.

Run after ingestion — ingestion never blocks waiting for embedding API calls.

Usage:
    python -m cadsentinel.etl.embedder
    python -m cadsentinel.etl.embedder --drawing-id 42
    python -m cadsentinel.etl.embedder --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Optional

from .db import get_connection

log = logging.getLogger(__name__)

EMBEDDING_MODEL  = os.environ.get("CADSENTINEL_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMS   = int(os.environ.get("CADSENTINEL_EMBEDDING_DIMS", "1536"))
BATCH_SIZE       = int(os.environ.get("CADSENTINEL_EMBED_BATCH_SIZE", "100"))
OPENAI_BASE_URL  = os.environ.get("OPENAI_BASE_URL")


def _get_openai_client():
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "openai package is required. Install it: pip install openai"
        ) from exc
    kwargs: dict = {}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return openai.OpenAI(**kwargs)


def _embed_batch(client, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def embed_pending(
    drawing_id: Optional[int] = None,
    dry_run: bool = False,
    batch_size: int = BATCH_SIZE,
) -> int:
    """
    Embed all pending text chunks (embedding IS NULL).
    Returns total number of chunks embedded.
    """
    client = None if dry_run else _get_openai_client()
    total_embedded = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            if drawing_id is not None:
                cur.execute(
                    """
                    SELECT id, chunk_text
                    FROM drawing_text_chunks
                    WHERE drawing_id = %s AND embedding IS NULL
                    ORDER BY id
                    """,
                    (drawing_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, chunk_text
                    FROM drawing_text_chunks
                    WHERE embedding IS NULL
                    ORDER BY id
                    """
                )
            rows = cur.fetchall()

    if not rows:
        log.info("No pending text chunks to embed.")
        return 0

    log.info("Found %d pending chunks%s",
             len(rows), f" for drawing_id={drawing_id}" if drawing_id else "")

    if dry_run:
        for row in rows[:10]:
            log.info("  [dry-run] id=%s: %s...", row["id"], row["chunk_text"][:80])
        if len(rows) > 10:
            log.info("  [dry-run] ... and %d more", len(rows) - 10)
        return 0

    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start: batch_start + batch_size]
        texts = [r["chunk_text"] for r in batch]
        ids   = [r["id"] for r in batch]

        try:
            vectors = _embed_batch(client, texts)
        except Exception:
            log.exception("Embedding API failed for batch at index %d", batch_start)
            continue

        with get_connection() as conn:
            with conn.cursor() as cur:
                for chunk_id, vector in zip(ids, vectors):
                    cur.execute(
                        """
                        UPDATE drawing_text_chunks
                        SET embedding      = %s::vector,
                            embedded_at    = NOW(),
                            embedded_model = %s
                        WHERE id = %s
                        """,
                        (str(vector), EMBEDDING_MODEL, chunk_id),
                    )
            conn.commit()

        total_embedded += len(batch)
        log.info("Embedded %d/%d chunks", total_embedded, len(rows))
        time.sleep(0.1)

    log.info("Embedding complete. Total: %d", total_embedded)
    return total_embedded


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="CADSentinel text chunk embedder")
    parser.add_argument("--drawing-id", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    count = embed_pending(
        drawing_id=args.drawing_id,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )
    print(f"Done. Embedded {count} chunks.")


if __name__ == "__main__":
    main()