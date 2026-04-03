"""
tests/test_retriever.py
-----------------------
Unit tests for the evidence retrieval engine.
No real database or embedding API required — uses mocks throughout.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from cadsentinel.etl.retriever import EvidenceRetriever, RetrievalError

DRAWING_ID    = 7
SPEC_RULE_ID  = 142
RULE_TEXT     = "All dimensions must be shown in inches unless otherwise specified."
CHECKER_PROMPT = "Check that all dimension entities use inches as the unit."

SAMPLE_RECIPE = {
    "source_types": ["note", "dimension", "title_block"],
    "top_k": 5,
    "keyword_filters": ["inch", "units"],
    "entity_filters": {"layers": ["DIM*", "ANNO*"]},
}


# ── Fixtures ─────────────────────────────────────────────────── #

def make_cur(rows_by_call=None):
    """
    Return a mock cursor.
    rows_by_call: list of return values for successive fetchall() / fetchone() calls.
    """
    cur = MagicMock()
    if rows_by_call:
        cur.fetchall.side_effect = [
            r for r in rows_by_call if isinstance(r, list)
        ]
        cur.fetchone.side_effect = [
            r for r in rows_by_call if not isinstance(r, list)
        ]
    else:
        cur.fetchall.return_value = []
        cur.fetchone.return_value = None
    return cur


def make_conn(cur):
    conn = MagicMock()
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


# ── Basic structure tests ─────────────────────────────────────── #

class TestEvidencePackageStructure:
    def test_package_has_required_keys(self):
        retriever = EvidenceRetriever(embed_for_vector_search=False)
        cur = make_cur()
        conn = make_conn(cur)

        with patch("cadsentinel.etl.retriever.get_connection") as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__ = MagicMock(return_value=False)

            package = retriever.retrieve(
                drawing_id=DRAWING_ID,
                spec_rule_id=SPEC_RULE_ID,
                normalized_rule_text=RULE_TEXT,
                retrieval_recipe=SAMPLE_RECIPE,
            )

        assert "spec_rule_id"     in package
        assert "drawing_id"       in package
        assert "retrieval_method" in package
        assert "query_text"       in package
        assert "evidence"         in package
        assert "evidence_count"   in package
        assert "truncated"        in package

    def test_spec_rule_id_and_drawing_id_correct(self):
        retriever = EvidenceRetriever(embed_for_vector_search=False)
        cur = make_cur()
        conn = make_conn(cur)

        with patch("cadsentinel.etl.retriever.get_connection") as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__ = MagicMock(return_value=False)

            package = retriever.retrieve(
                drawing_id=DRAWING_ID,
                spec_rule_id=SPEC_RULE_ID,
                normalized_rule_text=RULE_TEXT,
                retrieval_recipe=SAMPLE_RECIPE,
            )

        assert package["spec_rule_id"] == SPEC_RULE_ID
        assert package["drawing_id"]   == DRAWING_ID

    def test_method_is_metadata_when_no_vector_search(self):
        retriever = EvidenceRetriever(embed_for_vector_search=False)
        cur = make_cur()
        conn = make_conn(cur)

        with patch("cadsentinel.etl.retriever.get_connection") as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__ = MagicMock(return_value=False)

            package = retriever.retrieve(
                drawing_id=DRAWING_ID,
                spec_rule_id=SPEC_RULE_ID,
                normalized_rule_text=RULE_TEXT,
                retrieval_recipe=SAMPLE_RECIPE,
            )

        assert package["retrieval_method"] == "metadata"

    def test_method_is_hybrid_when_vector_search_enabled(self):
        retriever = EvidenceRetriever(embed_for_vector_search=True)
        cur = make_cur()
        # title_block fetchone returns None (not found), 
        # embedding count fetchone returns {"n": 0} (no embeddings)
        cur.fetchone.side_effect = [None, {"n": 0}]
        cur.fetchall.return_value = []
        conn = make_conn(cur)

        with patch("cadsentinel.etl.retriever.get_connection") as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__ = MagicMock(return_value=False)

            package = retriever.retrieve(
                drawing_id=DRAWING_ID,
                spec_rule_id=SPEC_RULE_ID,
                normalized_rule_text=RULE_TEXT,
                retrieval_recipe=SAMPLE_RECIPE,
            )

        assert package["retrieval_method"] == "hybrid"


# ── Query text selection ──────────────────────────────────────── #

class TestQueryTextSelection:
    def test_uses_normalized_text_when_no_checker_prompt(self):
        retriever = EvidenceRetriever(embed_for_vector_search=False)
        cur = make_cur()
        conn = make_conn(cur)

        with patch("cadsentinel.etl.retriever.get_connection") as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__ = MagicMock(return_value=False)

            package = retriever.retrieve(
                drawing_id=DRAWING_ID,
                spec_rule_id=SPEC_RULE_ID,
                normalized_rule_text=RULE_TEXT,
                retrieval_recipe=SAMPLE_RECIPE,
                checker_prompt=None,
            )

        assert package["query_text"] == RULE_TEXT

    def test_uses_checker_prompt_when_provided(self):
        retriever = EvidenceRetriever(embed_for_vector_search=False)
        cur = make_cur()
        conn = make_conn(cur)

        with patch("cadsentinel.etl.retriever.get_connection") as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__ = MagicMock(return_value=False)

            package = retriever.retrieve(
                drawing_id=DRAWING_ID,
                spec_rule_id=SPEC_RULE_ID,
                normalized_rule_text=RULE_TEXT,
                retrieval_recipe=SAMPLE_RECIPE,
                checker_prompt=CHECKER_PROMPT,
            )

        assert package["query_text"] == CHECKER_PROMPT


# ── Source type routing ───────────────────────────────────────── #

class TestSourceTypeRouting:
    def _run(self, source_types):
        retriever = EvidenceRetriever(embed_for_vector_search=False)
        cur = make_cur()
        conn = make_conn(cur)
        recipe = {**SAMPLE_RECIPE, "source_types": source_types}

        with patch("cadsentinel.etl.retriever.get_connection") as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__ = MagicMock(return_value=False)

            retriever.retrieve(
                drawing_id=DRAWING_ID,
                spec_rule_id=SPEC_RULE_ID,
                normalized_rule_text=RULE_TEXT,
                retrieval_recipe=recipe,
            )
        return cur

    def test_note_source_queries_drawing_entities(self):
        cur = self._run(["note"])
        sqls = [str(c.args[0]) for c in cur.execute.call_args_list]
        assert any("drawing_entities" in s for s in sqls)

    def test_dimension_source_queries_drawing_dimensions(self):
        cur = self._run(["dimension"])
        sqls = [str(c.args[0]) for c in cur.execute.call_args_list]
        assert any("drawing_dimensions" in s for s in sqls)

    def test_title_block_source_queries_drawing_title_block(self):
        cur = self._run(["title_block"])
        sqls = [str(c.args[0]) for c in cur.execute.call_args_list]
        assert any("drawing_title_block" in s for s in sqls)

    def test_layer_source_queries_drawing_layers(self):
        cur = self._run(["layer"])
        sqls = [str(c.args[0]) for c in cur.execute.call_args_list]
        assert any("drawing_layers" in s for s in sqls)

    def test_empty_source_types_queries_nothing(self):
        cur = self._run([])
        assert cur.execute.call_count == 0


# ── top_k trimming ────────────────────────────────────────────── #

class TestTopKTrimming:
    def test_evidence_trimmed_to_top_k(self):
        retriever = EvidenceRetriever(embed_for_vector_search=False)
        cur = MagicMock()

        # Return 10 text entity rows — more than top_k=3
        fake_rows = [
            {"entity_type": "TEXT", "handle": f"{i}", "layer": "A",
             "text_content": f"note {i}", "geometry_json": None}
            for i in range(10)
        ]
        cur.fetchall.return_value = fake_rows
        cur.fetchone.return_value = None
        conn = make_conn(cur)

        recipe = {**SAMPLE_RECIPE, "source_types": ["note"], "top_k": 3}

        with patch("cadsentinel.etl.retriever.get_connection") as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__ = MagicMock(return_value=False)

            package = retriever.retrieve(
                drawing_id=DRAWING_ID,
                spec_rule_id=SPEC_RULE_ID,
                normalized_rule_text=RULE_TEXT,
                retrieval_recipe=recipe,
            )

        assert package["evidence_count"] == 3
        assert package["truncated"] is True

    def test_not_truncated_when_under_top_k(self):
        retriever = EvidenceRetriever(embed_for_vector_search=False)
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"entity_type": "TEXT", "handle": "1", "layer": "A",
             "text_content": "note", "geometry_json": None}
        ]
        cur.fetchone.return_value = None
        conn = make_conn(cur)

        recipe = {**SAMPLE_RECIPE, "source_types": ["note"], "top_k": 8}

        with patch("cadsentinel.etl.retriever.get_connection") as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__ = MagicMock(return_value=False)

            package = retriever.retrieve(
                drawing_id=DRAWING_ID,
                spec_rule_id=SPEC_RULE_ID,
                normalized_rule_text=RULE_TEXT,
                retrieval_recipe=recipe,
            )

        assert package["truncated"] is False


# ── Deduplication ────────────────────────────────────────────── #

class TestDeduplication:
    def test_deduplicates_by_handle(self):
        retriever = EvidenceRetriever(embed_for_vector_search=False)

        items = [
            {"entity_handle": "AA", "source": "drawing_entities",
             "text": "note 1", "similarity_score": None},
            {"entity_handle": "AA", "source": "drawing_entities",
             "text": "note 1 duplicate", "similarity_score": None},
            {"entity_handle": "BB", "source": "drawing_entities",
             "text": "note 2", "similarity_score": None},
        ]

        result = retriever._deduplicate(items)
        handles = [r["entity_handle"] for r in result if r["entity_handle"]]
        assert len([h for h in handles if h == "AA"]) == 1
        assert len(result) == 2

    def test_prefers_vector_result_over_metadata(self):
        retriever = EvidenceRetriever(embed_for_vector_search=False)

        items = [
            {"entity_handle": "AA", "source": "drawing_entities",
             "text": "metadata", "similarity_score": None},
            {"entity_handle": "AA", "source": "drawing_text_chunks",
             "text": "vector", "similarity_score": 0.92},
        ]

        result = retriever._deduplicate(items)
        assert len(result) == 1
        assert result[0]["similarity_score"] == 0.92

    def test_none_handle_items_always_kept(self):
        retriever = EvidenceRetriever(embed_for_vector_search=False)

        items = [
            {"entity_handle": None, "source": "drawing_layers", "layer_name": "DIM"},
            {"entity_handle": None, "source": "drawing_layers", "layer_name": "ANNO"},
        ]

        result = retriever._deduplicate(items)
        assert len(result) == 2


# ── Ranking ───────────────────────────────────────────────────── #

class TestRanking:
    def test_ranked_by_similarity_score_descending(self):
        retriever = EvidenceRetriever(embed_for_vector_search=False)

        items = [
            {"entity_handle": "A", "similarity_score": 0.70},
            {"entity_handle": "B", "similarity_score": 0.95},
            {"entity_handle": "C", "similarity_score": 0.82},
        ]

        ranked = retriever._rank(items)
        scores = [r["similarity_score"] for r in ranked]
        assert scores == [0.95, 0.82, 0.70]

    def test_none_scores_go_to_end(self):
        retriever = EvidenceRetriever(embed_for_vector_search=False)

        items = [
            {"entity_handle": "A", "similarity_score": None},
            {"entity_handle": "B", "similarity_score": 0.88},
            {"entity_handle": "C", "similarity_score": None},
        ]

        ranked = retriever._rank(items)
        assert ranked[0]["similarity_score"] == 0.88
        assert ranked[1]["similarity_score"] is None
        assert ranked[2]["similarity_score"] is None


# ── Vector search skipped when no embeddings ─────────────────── #

class TestVectorSearchSkipping:
    def test_skips_vector_search_when_no_embeddings(self):
        retriever = EvidenceRetriever(embed_for_vector_search=True)
        cur = MagicMock()
        cur.fetchone.return_value = {"n": 0}  # no embeddings
        cur.fetchall.return_value = []
        conn = make_conn(cur)

        with patch("cadsentinel.etl.retriever.get_connection") as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__ = MagicMock(return_value=False)

            package = retriever.retrieve(
                drawing_id=DRAWING_ID,
                spec_rule_id=SPEC_RULE_ID,
                normalized_rule_text=RULE_TEXT,
                retrieval_recipe={**SAMPLE_RECIPE, "source_types": []},
            )

        assert package["evidence_count"] == 0

    def test_skips_vector_search_when_embed_disabled(self):
        """When embed_for_vector_search=False, the embedding API is never called."""
        retriever = EvidenceRetriever(embed_for_vector_search=False)
        cur = make_cur()
        conn = make_conn(cur)

        with patch("cadsentinel.etl.retriever.get_connection") as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__ = MagicMock(return_value=False)
            with patch.object(retriever, "_embed_text") as mock_embed:
                retriever.retrieve(
                    drawing_id=DRAWING_ID,
                    spec_rule_id=SPEC_RULE_ID,
                    normalized_rule_text=RULE_TEXT,
                    retrieval_recipe=SAMPLE_RECIPE,
                )
                mock_embed.assert_not_called()


# ── Layer pattern conversion ──────────────────────────────────── #

class TestLayerPatternConversion:
    def test_asterisk_converted_to_sql_wildcard(self):
        """DIM* in retrieval_recipe should become DIM% in SQL ILIKE."""
        retriever = EvidenceRetriever(embed_for_vector_search=False)
        cur = MagicMock()
        cur.fetchall.return_value = []
        cur.fetchone.return_value = None
        conn = make_conn(cur)

        with patch("cadsentinel.etl.retriever.get_connection") as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=conn)
            mock_gc.return_value.__exit__ = MagicMock(return_value=False)

            retriever.retrieve(
                drawing_id=DRAWING_ID,
                spec_rule_id=SPEC_RULE_ID,
                normalized_rule_text=RULE_TEXT,
                retrieval_recipe={
                    **SAMPLE_RECIPE,
                    "source_types": ["note"],
                    "entity_filters": {"layers": ["DIM*"]},
                },
            )

        all_params = []
        for c in cur.execute.call_args_list:
            if c.args and len(c.args) > 1:
                all_params.extend(list(c.args[1]))

        assert "DIM%" in all_params