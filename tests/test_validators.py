# test_validators.py

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
from cadsentinel.etl.validators.title_block     import TitleBlockValidator
from cadsentinel.etl.validators.layer_naming    import LayerNamingValidator, _any_match
from cadsentinel.etl.validators.dimension_units import DimensionUnitsValidator
from cadsentinel.etl.validators.revision_table  import RevisionTableValidator
from cadsentinel.etl.validators.model_code      import ModelCodeValidator
from cadsentinel.etl.validators.standard_notes  import StandardNotesValidator
from cadsentinel.etl.validators.cylinder_spec   import CylinderSpecValidator
from cadsentinel.etl.validators.jit_bore        import JITBoreValidator
from cadsentinel.etl.validators.jit_mount       import JITMountValidator


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
# ── Shared evidence builders for new validators ───────────────── #

def text_evidence(*texts):
    """Build an evidence package from one or more text strings."""
    items = [
        text_chunk_item(t, handle=f"T{i}")
        for i, t in enumerate(texts)
    ]
    return make_evidence(items)


def empty_evidence():
    return make_evidence([])


# ── ModelCodeValidator tests ──────────────────────────────────── #

class TestModelCodeValidator:
    v = ModelCodeValidator()

    def test_pass_when_full_model_code_present(self):
        ev = text_evidence("H-MT4-4X65X2-1-NC-S-V-S")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "pass"

    def test_pass_case_insensitive(self):
        ev = text_evidence("h-mt4-4x65x2-1-nc-s-v-s")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "pass"

    def test_pass_model_code_in_title_block(self):
        ev = make_evidence([title_block_item(attributes={"MATERIALS": "H-MT4-4X65X2-1-NC-S-V-S"})])
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "pass"

    def test_fail_when_partial_model_code_present(self):
        ev = text_evidence("H-MT4-4.0 something incomplete")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "fail"
        assert any(i["issue_type"] == "model_code_malformed" for i in result.issues)

    def test_needs_review_when_no_model_code(self):
        ev = text_evidence("GENERAL NOTES", "DIMENSIONS IN INCHES")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "needs_review"

    def test_needs_review_when_no_evidence(self):
        ev = empty_evidence()
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "needs_review"

    def test_evidence_used_contains_model_code(self):
        ev = text_evidence("H-MT4-4X65X2-1-NC-S-V-S")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert any("H-MT4" in str(e.get("value", "")) for e in result.evidence_used)

    def test_severity_propagated(self):
        ev = empty_evidence()
        result = self.v.validate(ev, {"severity_default": "critical"})
        assert result.severity == "critical"


# ── StandardNotesValidator tests ──────────────────────────────── #

class TestStandardNotesValidator:
    v = StandardNotesValidator()

    ALL_NOTES = (
        "-Dimensions are in Inches",
        "-[ ] Indicates Millimeters",
        "-Chamfer Size = .03x45 deg.",
        "-All Diameters within .002 TIR",
        "-Holes less than 0.500",
        "-Decimal: x.xxx 0.005",
        "-Decimal: x.xx 0.015",
        "-Angular: x 0.5",
    )

    def test_pass_when_all_notes_present(self):
        ev = text_evidence(*self.ALL_NOTES)
        result = self.v.validate(ev, {"severity_default": "medium"})
        assert result.pass_fail == "pass"

    def test_fail_when_one_note_missing(self):
        notes = list(self.ALL_NOTES)
        notes = [n for n in notes if "dimensions are in inches" not in n.lower()]
        ev = text_evidence(*notes)
        result = self.v.validate(ev, {"severity_default": "medium"})
        assert result.pass_fail == "fail"
        assert any("Dimension Units" in i["description"] for i in result.issues)

    def test_fail_reports_each_missing_note(self):
        ev = text_evidence("unrelated content only")
        result = self.v.validate(ev, {"severity_default": "medium"})
        assert result.pass_fail == "fail"
        assert len(result.issues) == 8

    def test_needs_review_when_no_evidence(self):
        ev = empty_evidence()
        result = self.v.validate(ev, {"severity_default": "medium"})
        assert result.pass_fail == "needs_review"

    def test_case_insensitive_matching(self):
        ev = text_evidence(*[n.upper() for n in self.ALL_NOTES])
        result = self.v.validate(ev, {"severity_default": "medium"})
        assert result.pass_fail == "pass"

    def test_notes_split_across_chunks_still_found(self):
        # Simulates the fragmented chunks we saw in G250.dwg
        ev = text_evidence(
            "-Dimensions are in Inches",
            "-[  ] Indicates Millimeters",
            "-Chamfer Size = .03x45 deg.",
            "-All Diameters",
            "within .002 TIR",
            "-Holes less than 0.500",
            "+0.003 -0.000",
            "-Decimal:   x.xxx",
            "0.005",
            "x.xx",
            "0.015",
            "-Angular:    x",
            "0.5",
        )
        result = self.v.validate(ev, {"severity_default": "medium"})
        assert result.pass_fail == "pass"

    def test_severity_propagated(self):
        ev = empty_evidence()
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.severity == "high"


# ── CylinderSpecValidator tests ───────────────────────────────── #

class TestCylinderSpecValidator:
    v = CylinderSpecValidator()

    def test_pass_when_all_entries_present(self):
        ev = text_evidence(
            "BORE - 3.250",
            "STROKE - 10",
            "ROD - 1.000",
            "PORTS: CEH - .500 NPT - PP1",
        )
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "pass"

    def test_fail_when_bore_missing(self):
        ev = text_evidence("STROKE - 10", "ROD - 1.000", "PORTS: CEH")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "fail"
        assert any("Bore" in i["description"] for i in result.issues)

    def test_fail_when_multiple_missing(self):
        ev = text_evidence("BORE - 3.250")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "fail"
        assert len(result.issues) == 3

    def test_fail_issue_type_correct(self):
        ev = text_evidence("STROKE - 10")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert all(i["issue_type"] == "cylinder_spec_missing" for i in result.issues)

    def test_needs_review_when_no_evidence(self):
        ev = empty_evidence()
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "needs_review"

    def test_case_insensitive_matching(self):
        ev = text_evidence("BORE - 3.250", "STROKE - 10", "ROD - 1.000", "PORTS: CEH")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "pass"

    def test_severity_propagated(self):
        ev = empty_evidence()
        result = self.v.validate(ev, {"severity_default": "critical"})
        assert result.severity == "critical"


# ── JITBoreValidator tests ────────────────────────────────────── #

class TestJITBoreValidator:
    v = JITBoreValidator()

    def test_pass_for_valid_bore_size(self):
        for bore in [1.5, 2.0, 2.5, 3.25, 4.0, 5.0, 6.0, 7.0, 8.0]:
            ev = text_evidence(f"BORE - {bore}")
            result = self.v.validate(ev, {"severity_default": "high"})
            assert result.pass_fail == "pass", f"Expected pass for bore {bore}"

    def test_fail_for_invalid_bore_size(self):
        ev = text_evidence("BORE - 3.0")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "fail"
        assert any(i["issue_type"] == "invalid_bore_size" for i in result.issues)

    def test_fail_issue_contains_valid_list(self):
        ev = text_evidence("BORE - 3.0")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert any("3.25" in i["description"] for i in result.issues)

    def test_needs_review_when_no_bore_entry(self):
        ev = text_evidence("STROKE - 10", "ROD - 1.000")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "needs_review"

    def test_needs_review_when_no_evidence(self):
        ev = empty_evidence()
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "needs_review"

    def test_bore_with_colon_separator(self):
        ev = text_evidence("BORE: 4.0")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "pass"

    def test_custom_valid_bore_sizes(self):
        ev = text_evidence("BORE - 3.0")
        result = self.v.validate(ev, {
            "severity_default": "high",
            "valid_bore_sizes": [3.0, 4.0, 5.0],
        })
        assert result.pass_fail == "pass"

    def test_evidence_used_contains_bore_value(self):
        ev = text_evidence("BORE - 4.0")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert any(e.get("value") == 4.0 for e in result.evidence_used)


# ── JITMountValidator tests ───────────────────────────────────── #

class TestJITMountValidator:
    v = JITMountValidator()

    def test_pass_for_valid_mount_code(self):
        ev = text_evidence("H-MT4-4X65X2-1-NC-S-V-S")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "pass"

    def test_pass_for_all_valid_mount_codes(self):
        valid_codes = [
            "MHF", "MCF", "ME5", "ME6", "MF1", "MF2", "MF5", "MF6",
            "MP1", "MP2", "MP3", "MPU3",
            "MS1", "MS2", "MS3", "MS4", "MS7",
            "MT1", "MT2", "MT4",
            "MX0", "MX1", "MX2", "MX3",
        ]
        for code in valid_codes:
            ev = text_evidence(f"H-{code}-4X65X2-1-NC-S-V-S")
            result = self.v.validate(ev, {"severity_default": "high"})
            assert result.pass_fail == "pass", f"Expected pass for mount code {code}"

    def test_fail_for_invalid_mount_code(self):
        ev = text_evidence("H-XYZ-4X65X2-1-NC-S-V-S")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "fail"
        assert any(i["issue_type"] == "invalid_mount_code" for i in result.issues)

    def test_fail_issue_contains_mount_code(self):
        ev = text_evidence("H-XYZ-4X65X2-1-NC-S-V-S")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert any("XYZ" in i["description"] for i in result.issues)

    def test_needs_review_when_no_model_code(self):
        ev = text_evidence("BORE - 4.0", "STROKE - 10")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "needs_review"

    def test_needs_review_when_no_evidence(self):
        ev = empty_evidence()
        result = self.v.validate(ev, {"severity_default": "high"})
        assert result.pass_fail == "needs_review"

    def test_custom_valid_mount_codes(self):
        ev = text_evidence("H-XYZ-4X65X2-1-NC-S-V-S")
        result = self.v.validate(ev, {
            "severity_default":  "high",
            "valid_mount_codes": ["XYZ", "ABC"],
        })
        assert result.pass_fail == "pass"

    def test_evidence_used_contains_mount_code(self):
        ev = text_evidence("H-MT4-4X65X2-1-NC-S-V-S")
        result = self.v.validate(ev, {"severity_default": "high"})
        assert any("MT4" in str(e.get("value", "")) for e in result.evidence_used)

    def test_severity_propagated(self):
        ev = empty_evidence()
        result = self.v.validate(ev, {"severity_default": "critical"})
        assert result.severity == "critical"