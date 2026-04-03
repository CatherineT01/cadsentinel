"""
tests/test_etl.py
-----------------
Unit tests for the CADSentinel ETL pipeline.
Tests all parsers and ingestor logic against synthetic dwg_inspect JSON.
No real DWG file or database required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cadsentinel.etl.parsers import (
    parse_layers,
    parse_blocks,
    parse_entities,
    parse_title_block,
    parse_dimensions,
    parse_text_chunks,
    extract_text_from_entity,
)
from cadsentinel.etl.ingestor import DwgIngestor, IngestionError


# ── Fixtures ─────────────────────────────────────────────────── #

SAMPLE_JSON = {
    "file": "test_drawing.dwg",
    "schema_version": "1.1.0",
    "libredwg_version": {"major": 0, "minor": 12},
    "header": {
        "version": 27,
        "codepage": 1252,
        "extents": {
            "model": {"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100},
            "paper": {"xmin": 0, "ymin": 0, "xmax": 210, "ymax": 297},
        },
    },
    "layers": [
        {"name": "0",           "flags": 0, "lineweight": 0},
        {"name": "DIMENSIONS",  "flags": 0, "lineweight": 25},
        {"name": "ANNOTATIONS", "flags": 4, "lineweight": 0},
    ],
    "blocks": [
        {"name": "TITLE_BLOCK",   "handle": "1A", "num_entities": 12, "entity_handles": ["2A"]},
        {"name": "*Model_Space",  "handle": "1F", "num_entities": 45, "entity_handles": []},
    ],
    "entities": [
        {
            "index": 10, "type": "TEXT", "category": "text",
            "layer": "ANNOTATIONS", "handle": "2A", "owner_handle": "1A",
            "text": "NOTE: ALL DIMENSIONS IN INCHES",
            "geometry": {"ins_pt": {"x": 10.0, "y": 20.0, "z": 0.0}, "height": 0.25},
        },
        {
            "index": 11, "type": "MTEXT", "category": "text",
            "layer": "ANNOTATIONS", "handle": "2B", "owner_handle": "1F",
            "text": "GENERAL NOTES:\n1. TOLERANCES +/-0.005",
            "geometry": {"ins_pt": {"x": 5.0, "y": 5.0, "z": 0.0}},
        },
        {
            "index": 20, "type": "DIMENSION_LINEAR", "category": "dimension",
            "layer": "DIMENSIONS", "handle": "3A", "owner_handle": "1F",
            "text": None, "value": 4.250,
            "geometry": {"text_position": {"x": 50.0, "y": 30.0, "z": 0.0}},
        },
        {
            "index": 30, "type": "LINE", "category": "curve",
            "layer": "0", "handle": "4A", "owner_handle": "1F",
            "geometry": {"start": {"x": 0}, "end": {"x": 10}},
        },
        {
            "index": 40, "type": "BLOCK", "category": "block",
            "layer": "0", "handle": "5A", "owner_handle": "1A",
        },
        {
            "index": 41, "type": "ENDBLK", "category": "block",
            "layer": "0", "handle": "5B", "owner_handle": "1A",
        },
    ],
    "title_block": {
        "found": True,
        "block_name": "TITLE_BLOCK",
        "handle": "1A",
        "layer": "0",
        "geometry": {"ins_pt": {"x": 0, "y": 0, "z": 0}},
        "candidates": [],
        "attributes": {
            "DRAWING_NO": "DWG-001",
            "REVISION": "A",
            "TITLE": "GEARBOX",
            "ENGINEER": "J.S",
            "DATE": "2024",
        },
    },
    "summary": {
        "num_objects": 100,
        "num_entities": 6,
        "entity_type_counts": {},
        "category_counts": {},
        "layer_counts": {},
    },
}


def make_cur():
    """Return a mock psycopg2 cursor."""
    cur = MagicMock()
    cur.fetchone.return_value = None
    return cur


# ── parse_layers ─────────────────────────────────────────────── #

class TestParseLayers:
    def test_inserts_all_valid_layers(self):
        cur = make_cur()
        assert parse_layers(cur, 1, SAMPLE_JSON["layers"]) == 3

    def test_skips_empty_name(self):
        cur = make_cur()
        assert parse_layers(cur, 1, [{"name": "", "flags": 0, "lineweight": 0}]) == 0

    def test_skips_missing_name(self):
        cur = make_cur()
        assert parse_layers(cur, 1, [{"flags": 0, "lineweight": 0}]) == 0

    def test_empty_list(self):
        cur = make_cur()
        assert parse_layers(cur, 1, []) == 0

    def test_continues_after_db_error(self):
        cur = make_cur()
        cur.execute.side_effect = [Exception("DB error"), None, None]
        assert parse_layers(cur, 1, SAMPLE_JSON["layers"]) == 2


# ── parse_blocks ─────────────────────────────────────────────── #

class TestParseBlocks:
    def test_inserts_all_blocks(self):
        cur = make_cur()
        assert parse_blocks(cur, 1, SAMPLE_JSON["blocks"]) == 2

    def test_skips_no_name(self):
        cur = make_cur()
        assert parse_blocks(cur, 1, [{"name": "", "handle": "AA", "num_entities": 0, "entity_handles": []}]) == 0


# ── parse_entities ───────────────────────────────────────────── #

class TestParseEntities:
    def test_skips_structural_types(self):
        cur = make_cur()
        # 6 total: TEXT, MTEXT, DIMENSION_LINEAR, LINE, BLOCK, ENDBLK
        # BLOCK + ENDBLK are structural → expect 4
        assert parse_entities(cur, 1, SAMPLE_JSON["entities"]) == 4

    def test_text_content_included(self):
        cur = make_cur()
        parse_entities(cur, 1, [SAMPLE_JSON["entities"][0]])
        values = cur.execute.call_args[0][1]
        assert "NOTE: ALL DIMENSIONS IN INCHES" in values

    def test_line_has_no_text(self):
        cur = make_cur()
        parse_entities(cur, 1, [SAMPLE_JSON["entities"][3]])
        values = cur.execute.call_args[0][1]
        # text_content is the 8th value (index 7)
        assert values[7] is None


# ── parse_title_block ────────────────────────────────────────── #

class TestParseTitleBlock:
    def test_inserts_found_block(self):
        cur = make_cur()
        parse_title_block(cur, 1, SAMPLE_JSON["title_block"])
        values = cur.execute.call_args[0][1]
        assert values[1] is True       # found
        assert values[2] == "TITLE_BLOCK"

    def test_high_confidence_for_many_attrs(self):
        cur = make_cur()
        parse_title_block(cur, 1, SAMPLE_JSON["title_block"])
        values = cur.execute.call_args[0][1]
        assert values[8] == 0.90       # detection_confidence

    def test_zero_confidence_not_found(self):
        cur = make_cur()
        parse_title_block(cur, 1, {"found": False, "attributes": {}, "candidates": []})
        values = cur.execute.call_args[0][1]
        assert values[8] == 0.0

    def test_handles_empty_dict(self):
        cur = make_cur()
        parse_title_block(cur, 1, {})  # must not raise
        assert cur.execute.call_count == 1


# ── parse_dimensions ─────────────────────────────────────────── #

class TestParseDimensions:
    def test_only_dimension_category(self):
        cur = make_cur()
        assert parse_dimensions(cur, 1, SAMPLE_JSON["entities"]) == 1

    def test_measured_value_extracted(self):
        cur = make_cur()
        parse_dimensions(cur, 1, [SAMPLE_JSON["entities"][2]])
        values = cur.execute.call_args[0][1]
        assert values[4] == 4.250      # measured_value

    def test_skips_non_dimension(self):
        cur = make_cur()
        non_dims = [e for e in SAMPLE_JSON["entities"] if e.get("category") != "dimension"]
        assert parse_dimensions(cur, 1, non_dims) == 0


# ── parse_text_chunks ────────────────────────────────────────── #

class TestParseTextChunks:
    def test_only_text_category(self):
        cur = make_cur()
        assert parse_text_chunks(cur, 1, SAMPLE_JSON["entities"]) == 2

    def test_skips_short_text(self):
        cur = make_cur()
        ents = [{"type": "TEXT", "category": "text", "handle": "ZZ",
                 "layer": "0", "text": "A", "geometry": {}}]
        assert parse_text_chunks(cur, 1, ents) == 0

    def test_skips_whitespace_only(self):
        cur = make_cur()
        ents = [{"type": "TEXT", "category": "text", "handle": "ZY",
                 "layer": "0", "text": "   ", "geometry": {}}]
        assert parse_text_chunks(cur, 1, ents) == 0

    def test_embedding_is_null(self):
        cur = make_cur()
        parse_text_chunks(cur, 1, [SAMPLE_JSON["entities"][0]])
        sql = cur.execute.call_args[0][0]
        assert "NULL" in sql

    def test_position_json_from_ins_pt(self):
        cur = make_cur()
        parse_text_chunks(cur, 1, [SAMPLE_JSON["entities"][0]])
        values = cur.execute.call_args[0][1]
        pos = json.loads(values[5])    # position_json
        assert pos["x"] == 10.0


# ── extract_text_from_entity ─────────────────────────────────── #

class TestExtractText:
    def test_returns_text_field(self):
        assert extract_text_from_entity({"text": "Hello"}) == "Hello"

    def test_falls_back_to_default_value(self):
        assert extract_text_from_entity({"default_value": "Default"}) == "Default"

    def test_empty_for_no_text(self):
        assert extract_text_from_entity({"category": "curve"}) == ""

    def test_strips_whitespace(self):
        assert extract_text_from_entity({"text": "  trimmed  "}) == "trimmed"


# ── DwgIngestor ──────────────────────────────────────────────── #

class TestDwgIngestor:
    def test_raises_if_inspector_missing(self, tmp_path):
        with pytest.raises(IngestionError, match="dwg_inspect binary not found"):
            DwgIngestor(inspector_path=tmp_path / "nonexistent")

    def test_raises_if_dwg_missing(self, tmp_path):
        inspector = tmp_path / "dwg_inspect"
        inspector.write_text("#!/bin/sh\n")
        inspector.chmod(0o755)
        ingestor = DwgIngestor(inspector_path=inspector)
        with pytest.raises(IngestionError, match="DWG file not found"):
            ingestor.ingest(tmp_path / "missing.dwg")

    def test_raises_on_nonzero_exit(self, tmp_path):
        inspector = tmp_path / "dwg_inspect.bat"
        inspector.write_text("@echo off\necho Error 1>&2\nexit /b 1\n")
        dwg = tmp_path / "bad.dwg"
        dwg.write_bytes(b"\x00" * 100)
        ingestor = DwgIngestor(inspector_path=inspector)
        with pytest.raises(IngestionError, match="dwg_inspect failed"):
            ingestor.ingest(dwg)

    def test_raises_on_invalid_json(self, tmp_path):
        inspector = tmp_path / "dwg_inspect.bat"
        inspector.write_text("@echo off\necho not json\n")
        dwg = tmp_path / "test.dwg"
        dwg.write_bytes(b"\x00" * 100)
        ingestor = DwgIngestor(inspector_path=inspector)
        with pytest.raises(IngestionError, match="not valid JSON"):
            ingestor.ingest(dwg)

    def test_raises_on_bad_schema_version(self, tmp_path):
        import json as _json
        bad = _json.dumps({"schema_version": "9.9.9", "entities": []})
        inspector = tmp_path / "dwg_inspect.bat"
        inspector.write_text(f"@echo off\necho {bad}\n")
        dwg = tmp_path / "test.dwg"
        dwg.write_bytes(b"\x00" * 100)
        ingestor = DwgIngestor(inspector_path=inspector)
        with pytest.raises(IngestionError, match="Unsupported dwg_inspect schema_version"):
            ingestor.ingest(dwg)