"""
tests/test_validators.py
-------------------------
Unit tests for all four deterministic validators.
No database, LLM, or file I/O required.
"""

from __future__ import annotations

import pytest

from cadsentinel.etl.validators.base import (
    ValidatorResult, pass_result, fail_result,
    needs_review_result, warning_result,
    make_issue, make_evidence_ref,
    get_title_block, get_layers, get_dimensions,
)
from cadsentinel.etl.validators.title_block    import TitleBlockValidator
from cadsentinel.etl.validators.layer_naming   import LayerNamingValidator, _any_match
from cadsentinel.etl.validators.dimension_units import DimensionUnitsValidator
from cadsentinel.etl.validators.revision_table  import RevisionTableValidator


# ── Evidence builders ─────────────────────────────────────────── #

def make_evidence(items: list[dict]) -> dict:
    return {"evidence": items, "evidence_count": len(items)}


def title_block_item(found=True, attributes=None, handle="TB1"):
    return {
        "source":     "drawing_title_block",
        "found":      found,
        "block_name": "TITLE_BLOCK",
        "entity_handle": handle,
        "attributes": attributes or {},
        "geometry":   {},
        "detection_confidence": 0.90,
    }


def layer_item(layer_name: str):
    return {
        "source":     "drawing_layers",
        "entity_handle": None,
        "layer_name": layer_name,
        "flags":      0,
        "lineweight": 0,
    }


def dim_item(user_text: str = "", handle: str = "D1", value: float = 1.0):
    return {
        "source":          "drawing_dimensions",
        "dim_type":        "DIMENSION_LINEAR",
        "entity_handle":   handle,
        "layer":           "DIM",
        "measured_value":  value,
        "user_text":       user_text,
        "text_position":   None,
        "geometry":        {},
        "similarity_score": None,
    }


def text_chunk_item(text: str, handle: str = "T1"):
    return {
        "source":        "drawing_text_chunks",
        "entity_handle": handle,
        "entity_type":   "MTEXT",
        "layer":         "ANNO",
        "text":          text,
        "position":      None,
        "similarity_score": None,
    }


def entity_item(etype: str, block_name: str = "", tag: str = "", handle: str = "E1"):
    return {
        "source":        "drawing_entities",
        "entity_type":   etype,
        "entity_handle": handle,
        "layer":         "0",
        "text":          None,
        "block_name":    block_name,
        "tag":           tag,
        "geometry":      {},
        "similarity_score": None,
    }


# ── Base contract tests ───────────────────────────────────────── #

class TestBaseContract:
    def test_pass_result_structure(self):
        r = pass_result(severity="high")
        assert r.pass_fail    == "pass"
        assert r.confidence   == 1.0
        assert r.severity     == "high"
        assert r.issue_summary == ""
        assert r.issues       == []

    def test_fail_result_structure(self):
        r = fail_result("Something failed", [make_issue("type", "desc")])
        assert r.pass_fail == "fail"
        assert len(r.issues) == 1

    def test_needs_review_result_structure(self):
        r = needs_review_result("No data")
        assert r.pass_fail == "needs_review"
        assert r.confidence == 0.0
        assert len(r.issues) == 1

    def test_warning_result_structure(self):
        r = warning_result("Minor issue", [make_issue("t", "d")])
        assert r.pass_fail == "warning"

    def test_to_dict_has_all_keys(self):
        r = pass_result()
        d = r.to_dict()
        for key in ["pass_fail", "confidence", "severity", "issue_summary", "issues", "evidence_used"]:
            assert key in d

    def test_validator_catches_unexpected_error(self):
        from cadsentinel.etl.validators.base import BaseValidator
        class BrokenValidator(BaseValidator):
            name = "broken"
            def _validate(self, evidence, rule_config):
                raise RuntimeError("unexpected!")
        v = BrokenValidator()
        result = v.validate({}, {})
        assert result.pass_fail == "needs_review"


# ── TitleBlockValidator tests ─────────────────────────────────── #

class TestTitleBlockValidator:
    v = TitleBlockValidator()

    def test_pass_when_all_required_fields_present(self):
        ev = make_evidence([title_block_item(attributes={
            "DRAWING_NO": "DWG-001",
            "REVISION":   "A",
            "TITLE":      "Gearbox Assembly",
        })])
        result = self.v.validate(ev, {
            "required_fields":  ["DRAWING_NO", "REVISION", "TITLE"],
            "severity_default": "high",
        })
        assert result.pass_fail == "pass"

    def test_fail_when_required_field_missing(self):
        ev = make_evidence([title_block_item(attributes={
            "DRAWING_NO": "DWG-001",
        })])
        result = self.v.validate(ev, {
            "required_fields":  ["DRAWING_NO", "REVISION"],
            "severity_default": "high",
        })
        assert result.pass_fail == "fail"
        assert any("REVISION" in i["description"] for i in result.issues)

    def test_fail_when_required_field_empty(self):
        ev = make_evidence([title_block_item(attributes={
            "DRAWING_NO": "DWG-001",
            "REVISION":   "",
        })])
        result = self.v.validate(ev, {
            "required_fields":  ["DRAWING_NO", "REVISION"],
            "severity_default": "high",
        })
        assert result.pass_fail == "fail"
        assert any(i["issue_type"] == "title_block_field_empty" for i in result.issues)

    def test_fail_when_no_title_block_found(self):
        ev = make_evidence([title_block_item(found=False)])
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "fail"
        assert any(i["issue_type"] == "title_block_missing" for i in result.issues)

    def test_needs_review_when_no_tb_evidence(self):
        ev = make_evidence([])
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "needs_review"

    def test_pass_with_no_required_fields_configured(self):
        ev = make_evidence([title_block_item(attributes={"DRAWING_NO": "X"})])
        result = self.v.validate(ev, {"required_fields": []})
        assert result.pass_fail == "pass"

    def test_case_insensitive_field_lookup(self):
        ev = make_evidence([title_block_item(attributes={"drawing_no": "DWG-001"})])
        result = self.v.validate(ev, {"required_fields": ["DRAWING_NO"]})
        assert result.pass_fail == "pass"

    def test_multiple_missing_fields_reported(self):
        ev = make_evidence([title_block_item(attributes={})])
        result = self.v.validate(ev, {
            "required_fields": ["DRAWING_NO", "REVISION", "TITLE"],
        })
        assert result.pass_fail == "fail"
        assert len(result.issues) == 3

    def test_severity_propagated_to_result(self):
        ev = make_evidence([title_block_item(found=False)])
        result = self.v.validate(ev, {"severity_default": "critical"})
        assert result.severity == "critical"


# ── LayerNamingValidator tests ────────────────────────────────── #

class TestLayerNamingValidator:
    v = LayerNamingValidator()

    def test_pass_when_required_pattern_matched(self):
        ev = make_evidence([
            layer_item("DIM-LINEAR"),
            layer_item("ANNO-TEXT"),
            layer_item("0"),
        ])
        result = self.v.validate(ev, {"required_patterns": ["DIM*"]})
        assert result.pass_fail == "pass"

    def test_fail_when_required_pattern_not_matched(self):
        ev = make_evidence([layer_item("0"), layer_item("OBJECTS")])
        result = self.v.validate(ev, {"required_patterns": ["DIM*"]})
        assert result.pass_fail == "fail"
        assert any(i["issue_type"] == "required_layer_missing" for i in result.issues)

    def test_fail_when_forbidden_layer_present(self):
        ev = make_evidence([layer_item("TEMP"), layer_item("0")])
        result = self.v.validate(ev, {"forbidden_patterns": ["TEMP*"]})
        assert result.pass_fail == "fail"
        assert any(i["issue_type"] == "forbidden_layer_present" for i in result.issues)

    def test_pass_when_no_forbidden_layers(self):
        ev = make_evidence([layer_item("DIM"), layer_item("0")])
        result = self.v.validate(ev, {"forbidden_patterns": ["TEMP*"]})
        assert result.pass_fail == "pass"

    def test_needs_review_when_no_layer_evidence(self):
        ev = make_evidence([])
        result = self.v.validate(ev, {"required_patterns": ["DIM*"]})
        assert result.pass_fail == "needs_review"

    def test_pattern_matching_case_insensitive(self):
        ev = make_evidence([layer_item("dim-linear")])
        result = self.v.validate(ev, {"required_patterns": ["DIM*"]})
        assert result.pass_fail == "pass"

    def test_multiple_required_patterns_all_must_match(self):
        ev = make_evidence([layer_item("DIM-1"), layer_item("0")])
        result = self.v.validate(ev, {
            "required_patterns": ["DIM*", "ANNO*"],
            "require_all": True,
        })
        assert result.pass_fail == "fail"
        assert any("ANNO*" in i["description"] for i in result.issues)

    def test_wildcard_star_matches_any_suffix(self):
        assert _any_match(["DIM-LINEAR", "DIM-ANGULAR"], "DIM*") is True
        assert _any_match(["OBJECTS", "0"], "DIM*") is False

    def test_exact_match_works(self):
        ev = make_evidence([layer_item("BORDER")])
        result = self.v.validate(ev, {"required_patterns": ["BORDER"]})
        assert result.pass_fail == "pass"

    def test_both_required_and_forbidden_checked(self):
        ev = make_evidence([layer_item("DIM-1"), layer_item("TEMP")])
        result = self.v.validate(ev, {
            "required_patterns":  ["DIM*"],
            "forbidden_patterns": ["TEMP*"],
        })
        assert result.pass_fail == "fail"
        assert any(i["issue_type"] == "forbidden_layer_present" for i in result.issues)


# ── DimensionUnitsValidator tests ─────────────────────────────── #

class TestDimensionUnitsValidator:
    v = DimensionUnitsValidator()

    def test_pass_when_title_block_declares_correct_unit(self):
        ev = make_evidence([title_block_item(attributes={"UNITS": "INCHES"})])
        result = self.v.validate(ev, {"required_unit": "inches"})
        assert result.pass_fail == "pass"

    def test_fail_when_title_block_declares_wrong_unit(self):
        ev = make_evidence([title_block_item(attributes={"UNITS": "MM"})])
        result = self.v.validate(ev, {"required_unit": "inches"})
        assert result.pass_fail == "fail"
        assert any(i["issue_type"] == "unit_conflict" for i in result.issues)

    def test_pass_when_note_declares_correct_unit(self):
        ev = make_evidence([
            text_chunk_item("ALL DIMENSIONS IN INCHES UNLESS OTHERWISE SPECIFIED")
        ])
        result = self.v.validate(ev, {"required_unit": "inches"})
        assert result.pass_fail == "pass"

    def test_fail_when_dimension_text_shows_wrong_unit(self):
        ev = make_evidence([dim_item(user_text="25.4mm")])
        result = self.v.validate(ev, {"required_unit": "inches"})
        assert result.pass_fail == "fail"

    def test_warning_when_no_unit_info_found(self):
        ev = make_evidence([dim_item(user_text=""), dim_item(user_text="")])
        result = self.v.validate(ev, {"required_unit": "inches"})
        assert result.pass_fail in ("warning", "needs_review")

    def test_needs_review_when_no_evidence(self):
        ev = make_evidence([])
        result = self.v.validate(ev, {"required_unit": "inches"})
        assert result.pass_fail == "needs_review"

    def test_allow_mixed_returns_warning_not_fail(self):
        ev = make_evidence([
            dim_item(user_text="25.4mm"),
            text_chunk_item("ALL DIMENSIONS IN INCHES"),
        ])
        result = self.v.validate(ev, {
            "required_unit": "inches",
            "allow_mixed":   True,
        })
        assert result.pass_fail == "warning"

    def test_mm_unit_check(self):
        ev = make_evidence([title_block_item(attributes={"UNITS": "MM"})])
        result = self.v.validate(ev, {"required_unit": "mm"})
        assert result.pass_fail == "pass"

    def test_inch_symbol_detected(self):
        ev = make_evidence([text_chunk_item('TOLERANCES: +/- 0.005"')])
        result = self.v.validate(ev, {"required_unit": "inches"})
        assert result.pass_fail in ("pass", "warning")


# ── RevisionTableValidator tests ──────────────────────────────── #

class TestRevisionTableValidator:
    v = RevisionTableValidator()

    def test_pass_when_rev_table_block_found(self):
        ev = make_evidence([
            entity_item("INSERT", block_name="REV_TABLE", handle="E1"),
        ])
        result = self.v.validate(ev, {"severity_default": "medium"})
        assert result.pass_fail == "pass"

    def test_pass_when_revision_block_name_matches_pattern(self):
        ev = make_evidence([
            entity_item("INSERT", block_name="REVISION_HISTORY", handle="E2"),
        ])
        result = self.v.validate(ev, {})
        assert result.pass_fail == "pass"

    def test_pass_when_revision_in_title_block(self):
        ev = make_evidence([
            title_block_item(attributes={"REVISION": "A", "REV_DATE": "2024-01-15"}),
        ])
        result = self.v.validate(ev, {"min_revisions": 1})
        assert result.pass_fail == "pass"

    def test_pass_when_rev_attrib_entity_found(self):
        ev = make_evidence([
            entity_item("ATTRIB", tag="REVISION", handle="E3"),
        ])
        result = self.v.validate(ev, {})
        assert result.pass_fail == "pass"

    def test_fail_when_no_revision_info_found(self):
        ev = make_evidence([
            layer_item("0"),
            layer_item("DIM"),
        ])
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "fail"
        assert any(i["issue_type"] == "revision_table_missing" for i in result.issues)

    def test_needs_review_when_no_evidence_at_all(self):
        ev = {"evidence": [], "evidence_count": 0}
        result = self.v.validate(ev, {})
        assert result.pass_fail == "needs_review"

    def test_custom_block_pattern_respected(self):
        ev = make_evidence([
            entity_item("INSERT", block_name="ECN_LOG", handle="E4"),
        ])
        result = self.v.validate(ev, {
            "required_block_patterns": ["ECN*"],
        })
        assert result.pass_fail == "pass"

    def test_pattern_matching_case_insensitive(self):
        ev = make_evidence([
            entity_item("INSERT", block_name="rev_table", handle="E5"),
        ])
        result = self.v.validate(ev, {})
        assert result.pass_fail == "pass"

    def test_severity_propagated(self):
        ev = make_evidence([layer_item("0")])
        result = self.v.validate(ev, {"severity_default": "critical"})
        assert result.severity == "critical"