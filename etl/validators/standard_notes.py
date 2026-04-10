# standard_notes.py

"""
cadsentinel.etl.validators.standard_notes
-------------------------------------------
Validates that all eight JIT standard notes are present in the drawing.

Required notes (from JIT H Series Drawing Standard section 2):
    2.1  Dimensions are in Inches
    2.2  [ ] Indicates Millimeters
    2.3  Chamfer Size = .03"x45 deg.
    2.4  All Diameters within .002" TIR
    2.5  Holes less than 0.500"
    2.6  Decimal x.xxx tolerance
    2.7  Decimal x.xx tolerance
    2.8  Angular tolerance

Notes are checked against the combined text of all drawing text chunks
and entities. Matching is case-insensitive and tolerates minor whitespace
variation. AutoCAD control codes (%%p, %%d) are substituted before matching.

Returns:
    pass         — all required notes present
    fail         — one or more required notes missing
    needs_review — no text evidence available
"""

from __future__ import annotations

from .base import (
    BaseValidator,
    ValidatorResult,
    collect_all_text,
    make_issue,
    make_evidence_ref,
    pass_result,
    fail_result,
    needs_review_result,
)

# Each entry is (spec_code, human_label, search_substring)
# search_substring is checked against the normalized combined text
_REQUIRED_NOTES: list[tuple[str, str, str]] = [
    ("2.1", "Dimension Units",          "dimensions are in inches"),
    ("2.2", "Metric Brackets",          "indicates millimeters"),
    ("2.3", "Chamfer Size",             "chamfer size"),
    ("2.4", "Diameter Concentricity",   "all diameters"),
    ("2.5", "Hole Tolerance",           "holes less than"),
    ("2.6", "Three Place Decimal",      "x.xxx"),
    ("2.7", "Two Place Decimal",        "x.xx"),
    ("2.8", "Angular Tolerance",        "angular"),
]


class StandardNotesValidator(BaseValidator):
    """
    Checks that all eight JIT standard notes are present in the drawing.

    rule_config keys:
        severity_default (str): severity level, default 'medium'
        required_notes (list):  optional override list of (code, label, substring)
                                tuples — uses JIT defaults if not provided
    """

    name = "standard_notes"

    def _validate(
        self,
        evidence:    dict,
        rule_config: dict,
    ) -> ValidatorResult:

        severity = rule_config.get("severity_default", "medium")

        combined = collect_all_text(evidence)

        if not combined:
            return needs_review_result(
                reason   = "No text evidence available to check standard notes.",
                severity = severity,
            )

        single_note = rule_config.get("required_note")
        alt_note    = rule_config.get("alt_note")

        if single_note:
            spec_code, label, substring = str(single_note[0]), str(single_note[1]), str(single_note[2])
            found = substring.lower() in combined
            if not found and alt_note:
                found = str(alt_note).lower() in combined
            if found:
                return pass_result(
                    severity      = severity,
                    evidence_used = [make_evidence_ref(
                        source = "text_scan",
                        ref    = spec_code,
                        value  = label,
                    )],
                )
            return fail_result(
                issue_summary = f"1 standard note(s) missing: {label}.",
                issues        = [make_issue(
                    issue_type    = "standard_note_missing",
                    description   = (
                        f"Required standard note '{label}' (rule {spec_code}) "
                        f"not found in drawing. Expected text containing: '{substring}'."
                    ),
                    severity      = severity,
                    suggested_fix = f"Add the standard note for {label} to the drawing notes block.",
                )],
                severity      = severity,
                evidence_used = [],
            )

        missing_issues: list[dict] = []
        found_notes:    list[dict] = []

        for entry in _REQUIRED_NOTES:
            spec_code, label, substring = entry[0], entry[1], entry[2]
            if substring.lower() in combined:
                found_notes.append(make_evidence_ref(
                    source = "text_scan",
                    ref    = spec_code,
                    value  = label,
                ))
            else:
                missing_issues.append(make_issue(
                    issue_type    = "standard_note_missing",
                    description   = (
                        f"Required standard note '{label}' (rule {spec_code}) "
                        f"not found in drawing. Expected text containing: '{substring}'."
                    ),
                    severity      = severity,
                    suggested_fix = f"Add the standard note for {label} to the drawing notes block.",
                ))

        if missing_issues:
            missing_labels = [i["description"].split("'")[1] for i in missing_issues]
            return fail_result(
                issue_summary = f"{len(missing_issues)} standard note(s) missing: {', '.join(missing_labels)}.",
                issues        = missing_issues,
                severity      = severity,
                evidence_used = found_notes,
            )

        return pass_result(
            severity      = severity,
            evidence_used = found_notes,
        )