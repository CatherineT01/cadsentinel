"""
cadsentinel.etl.compatibility.compatibility_checker
-----------------------------------------------------
Compares drawings in a set against each other to find
dimensional discrepancies.

Logic:
    1. Find assembly drawing(s) in the set
    2. Extract key specs from assembly (bore, stroke, rod, ports)
    3. Compare each component drawing against assembly specs
    4. Flag mismatches and missing specs

Checks performed:
    Assembly vs Rod:        rod diameter match
    Assembly vs Barrel:     bore diameter match
    Assembly vs Gland:      inner bore should match rod diameter
    Assembly vs Piston:     outer diameter should match bore
    Assembly vs Cap/Rod End Head: port sizes should match
    All drawings:           drawing type consistency
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .spec_extractor import DrawingSpecs

log = logging.getLogger(__name__)

TOLERANCE = 0.010  # inch tolerance for dimensional comparisons


@dataclass
class CompatibilityIssue:
    drawing_id:   int
    filename:     str
    issue_type:   str
    description:  str
    severity:     str   # 'critical', 'warning', 'info'
    expected:     str = ""
    found:        str = ""


@dataclass
class CompatibilityResult:
    folder_path:      str
    drawings_checked: int
    assembly_count:   int
    issues:           list[CompatibilityIssue] = field(default_factory=list)
    groups:           list[dict]               = field(default_factory=list)
    summary:          str                      = ""


def check_compatibility(
    specs_list:  list[DrawingSpecs],
    folder_path: str = "",
) -> CompatibilityResult:
    """
    Check compatibility of a set of drawings.

    Args:
        specs_list:  List of DrawingSpecs extracted from drawings
        folder_path: Source folder path for reporting

    Returns:
        CompatibilityResult with issues and summary
    """
    result = CompatibilityResult(
        folder_path      = folder_path,
        drawings_checked = len(specs_list),
        assembly_count   = 0,
    )

    if not specs_list:
        result.summary = "No drawings to check."
        return result

    # Separate assemblies from components
    assemblies  = [s for s in specs_list if s.drawing_type == "assembly"]
    components  = [s for s in specs_list if s.drawing_type != "assembly"]

    result.assembly_count = len(assemblies)

    if not assemblies:
        result.summary = (
            f"No assembly drawing found in folder. "
            f"Checked {len(components)} component drawing(s) for individual compliance only. "
            f"Dimensional compatibility check requires at least one assembly drawing."
        )
        result.issues.append(CompatibilityIssue(
            drawing_id  = 0,
            filename    = folder_path,
            issue_type  = "no_assembly",
            description = "No assembly drawing found in this folder. Cannot perform dimensional compatibility check.",
            severity    = "warning",
        ))
        return result

    # Group components by best matching assembly
    groups = _group_by_assembly(assemblies, components)
    result.groups = groups

    # Check each group
    all_issues: list[CompatibilityIssue] = []

    for group in groups:
        assembly   = group["assembly"]
        group_components = group["components"]
        unmatched  = group.get("unmatched", [])

        # Check each component against the assembly
        for component in group_components:
            issues = _check_component(assembly, component)
            all_issues.extend(issues)

        # Report unmatched components
        for component in unmatched:
            all_issues.append(CompatibilityIssue(
                drawing_id  = component.drawing_id,
                filename    = component.filename,
                issue_type  = "unmatched_component",
                description = (
                    f"Could not match this {component.drawing_type} drawing "
                    f"to any assembly in the folder. Manual review required."
                ),
                severity    = "info",
            ))

    result.issues = all_issues

    # Generate summary
    critical = sum(1 for i in all_issues if i.severity == "critical")
    warnings = sum(1 for i in all_issues if i.severity == "warning")
    info     = sum(1 for i in all_issues if i.severity == "info")

    result.summary = (
        f"Checked {len(specs_list)} drawing(s) across {len(assemblies)} assembly group(s). "
        f"Found {critical} critical issue(s), {warnings} warning(s), {info} informational note(s)."
    )

    return result


def _group_by_assembly(
    assemblies: list[DrawingSpecs],
    components: list[DrawingSpecs],
) -> list[dict]:
    """Group component drawings with their matching assembly."""

    if len(assemblies) == 1:
        # Simple case — one assembly, all components belong to it
        return [{
            "assembly":   assemblies[0],
            "components": components,
            "unmatched":  [],
        }]

    # Multiple assemblies — match components by bore/stroke/rod
    groups = [{"assembly": a, "components": [], "unmatched": []} for a in assemblies]

    for component in components:
        best_group  = None
        best_score  = -1

        for group in groups:
            assembly = group["assembly"]
            score    = _match_score(assembly, component)
            if score > best_score:
                best_score = score
                best_group = group

        if best_group is not None and best_score > 0:
            best_group["components"].append(component)
        else:
            # Add to unmatched of first group
            groups[0]["unmatched"].append(component)

    return groups


def _match_score(assembly: DrawingSpecs, component: DrawingSpecs) -> int:
    """Score how well a component matches an assembly."""
    score = 0
    if assembly.bore and component.bore:
        if abs(assembly.bore - component.bore) <= TOLERANCE:
            score += 3
    if assembly.stroke and component.stroke:
        if abs(assembly.stroke - component.stroke) <= TOLERANCE:
            score += 2
    if assembly.rod and component.rod:
        if abs(assembly.rod - component.rod) <= TOLERANCE:
            score += 2
    return score


def _check_component(
    assembly:  DrawingSpecs,
    component: DrawingSpecs,
) -> list[CompatibilityIssue]:
    """Check a component drawing against assembly specs."""
    issues = []
    dtype  = component.drawing_type

    if dtype == "barrel":
        issues.extend(_check_barrel(assembly, component))
    elif dtype == "rod":
        issues.extend(_check_rod(assembly, component))
    elif dtype == "gland":
        issues.extend(_check_gland(assembly, component))
    elif dtype == "piston":
        issues.extend(_check_piston(assembly, component))
    elif dtype in ("rod_end_head", "cap_end_head"):
        issues.extend(_check_head(assembly, component))
    else:
        # Generic part — check what we can
        issues.extend(_check_generic(assembly, component))

    return issues


def _check_barrel(
    assembly:  DrawingSpecs,
    component: DrawingSpecs,
) -> list[CompatibilityIssue]:
    """Barrel inner bore should match assembly bore."""
    issues = []
    if assembly.bore and component.bore:
        if abs(assembly.bore - component.bore) > TOLERANCE:
            issues.append(CompatibilityIssue(
                drawing_id  = component.drawing_id,
                filename    = component.filename,
                issue_type  = "bore_mismatch",
                description = (
                    f"Barrel bore diameter does not match assembly specification. "
                    f"This may cause the piston to not fit correctly."
                ),
                severity    = "critical",
                expected    = f"{assembly.bore}\"",
                found       = f"{component.bore}\"",
            ))
    elif assembly.bore and component.bore is None:
        issues.append(CompatibilityIssue(
            drawing_id  = component.drawing_id,
            filename    = component.filename,
            issue_type  = "bore_missing",
            description = "Barrel drawing does not specify bore diameter. Cannot verify compatibility with assembly.",
            severity    = "warning",
            expected    = f"{assembly.bore}\"",
            found       = "not found",
        ))
    return issues


def _check_rod(
    assembly:  DrawingSpecs,
    component: DrawingSpecs,
) -> list[CompatibilityIssue]:
    """Rod diameter should match assembly rod spec."""
    issues = []
    if assembly.rod and component.rod:
        if abs(assembly.rod - component.rod) > TOLERANCE:
            issues.append(CompatibilityIssue(
                drawing_id  = component.drawing_id,
                filename    = component.filename,
                issue_type  = "rod_mismatch",
                description = (
                    f"Rod diameter does not match assembly specification. "
                    f"This may cause sealing or fit issues."
                ),
                severity    = "critical",
                expected    = f"{assembly.rod}\"",
                found       = f"{component.rod}\"",
            ))
    elif assembly.rod and component.rod is None:
        issues.append(CompatibilityIssue(
            drawing_id  = component.drawing_id,
            filename    = component.filename,
            issue_type  = "rod_missing",
            description = "Rod drawing does not specify rod diameter. Cannot verify compatibility with assembly.",
            severity    = "warning",
            expected    = f"{assembly.rod}\"",
            found       = "not found",
        ))
    return issues


def _check_gland(
    assembly:  DrawingSpecs,
    component: DrawingSpecs,
) -> list[CompatibilityIssue]:
    """Gland inner bore should match assembly rod diameter."""
    issues = []
    if assembly.rod and component.bore:
        if abs(assembly.rod - component.bore) > TOLERANCE:
            issues.append(CompatibilityIssue(
                drawing_id  = component.drawing_id,
                filename    = component.filename,
                issue_type  = "gland_bore_mismatch",
                description = (
                    f"Gland inner bore does not match assembly rod diameter. "
                    f"The rod will not seal correctly through the gland."
                ),
                severity    = "critical",
                expected    = f"{assembly.rod}\" (rod diameter)",
                found       = f"{component.bore}\"",
            ))
    return issues


def _check_piston(
    assembly:  DrawingSpecs,
    component: DrawingSpecs,
) -> list[CompatibilityIssue]:
    """Piston outer diameter should match assembly bore."""
    issues = []
    if assembly.bore and component.bore:
        diff = abs(assembly.bore - component.bore)
        if diff > TOLERANCE and diff < 0.5:
            # Small difference — likely a clearance fit, flag as warning
            issues.append(CompatibilityIssue(
                drawing_id  = component.drawing_id,
                filename    = component.filename,
                issue_type  = "piston_fit_warning",
                description = (
                    f"Piston diameter differs from assembly bore by {diff:.3f}\". "
                    f"Verify this is within acceptable clearance tolerance."
                ),
                severity    = "warning",
                expected    = f"~{assembly.bore}\" (assembly bore)",
                found       = f"{component.bore}\"",
            ))
        elif diff >= 0.5:
            issues.append(CompatibilityIssue(
                drawing_id  = component.drawing_id,
                filename    = component.filename,
                issue_type  = "piston_bore_mismatch",
                description = (
                    f"Piston diameter significantly differs from assembly bore. "
                    f"Piston will not fit in the cylinder."
                ),
                severity    = "critical",
                expected    = f"{assembly.bore}\"",
                found       = f"{component.bore}\"",
            ))
    return issues


def _check_head(
    assembly:  DrawingSpecs,
    component: DrawingSpecs,
) -> list[CompatibilityIssue]:
    """Check head port sizes against assembly port specs."""
    issues = []
    if assembly.ports and component.ports:
        assy_ports = sorted(assembly.ports)
        comp_ports = sorted(component.ports)
        if assy_ports and comp_ports:
            if abs(assy_ports[0] - comp_ports[0]) > TOLERANCE:
                issues.append(CompatibilityIssue(
                    drawing_id  = component.drawing_id,
                    filename    = component.filename,
                    issue_type  = "port_size_mismatch",
                    description = (
                        f"Head port size does not match assembly port specification. "
                        f"Port connections may not fit."
                    ),
                    severity    = "warning",
                    expected    = f"{assy_ports[0]}\"",
                    found       = f"{comp_ports[0]}\"",
                ))
    return issues


def _check_generic(
    assembly:  DrawingSpecs,
    component: DrawingSpecs,
) -> list[CompatibilityIssue]:
    """Generic check — flag if key dimensions are present and mismatched."""
    issues = []
    if assembly.bore and component.bore:
        if abs(assembly.bore - component.bore) > TOLERANCE:
            issues.append(CompatibilityIssue(
                drawing_id  = component.drawing_id,
                filename    = component.filename,
                issue_type  = "dimension_mismatch",
                description = "Bore dimension in this drawing does not match the assembly specification.",
                severity    = "warning",
                expected    = f"{assembly.bore}\"",
                found       = f"{component.bore}\"",
            ))
    return issues