"""Deterministic diff for Postgres EXPLAIN (ANALYZE, FORMAT JSON) plans.

We pre-chew the raw plan into a structured set of findings — node-type flips,
row-estimate misses, sort/hash spills, parallelism changes, top contributors to
runtime — and hand only that *summary* to the LLM. This keeps prompt size
manageable, prevents the model from inventing numbers, and gives it real
quantitative signal to reason about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Plan loading and normalization
# ---------------------------------------------------------------------------


def load_plan(path: str | Path) -> dict[str, Any]:
    """Load an EXPLAIN (ANALYZE, FORMAT JSON) document.

    Accepts the canonical Postgres shape — a single-element list containing
    {"Plan": ..., "Execution Time": ..., ...} — and a few common variants.
    """
    text = Path(path).read_text()
    data = json.loads(text)
    if isinstance(data, list):
        if not data:
            raise ValueError(f"{path}: empty plan array")
        data = data[0]
    if "Plan" not in data:
        raise ValueError(f"{path}: missing 'Plan' key — is this EXPLAIN FORMAT JSON?")
    return data


def walk(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield every node in the plan tree (depth-first, root first)."""
    yield node
    for child in node.get("Plans", []) or []:
        yield from walk(child)


def node_signature(node: dict[str, Any]) -> str:
    """Stable identity for a plan node, used to align baseline ↔ incident.

    We key on (Node Type, Relation Name / Index Name / Subplan Name) so the
    same logical step lines up even if its position in the tree shifts.
    """
    parts = [node.get("Node Type", "?")]
    for k in ("Relation Name", "Index Name", "Subplan Name", "CTE Name"):
        if v := node.get(k):
            parts.append(f"{k[:3]}:{v}")
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Findings — the structured signal we hand to Gemma
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """One concrete, named issue extracted from the plan diff."""

    kind: str              # "scan_method_flip", "row_estimate_miss", ...
    severity: str          # "critical", "warning", "info"
    summary: str           # one-sentence human-readable line
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiffReport:
    baseline_path: str
    incident_path: str
    baseline_runtime_ms: float | None
    incident_runtime_ms: float | None
    baseline_planning_ms: float | None
    incident_planning_ms: float | None
    query: str | None
    findings: list[Finding] = field(default_factory=list)
    hot_nodes: list[dict[str, Any]] = field(default_factory=list)

    def slowdown_factor(self) -> float | None:
        if self.baseline_runtime_ms and self.incident_runtime_ms:
            return self.incident_runtime_ms / self.baseline_runtime_ms
        return None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["slowdown_factor"] = self.slowdown_factor()
        return d


# ---------------------------------------------------------------------------
# The diff itself
# ---------------------------------------------------------------------------

# Scan methods that almost always indicate a regression when they replace
# an Index Scan / Index Only Scan / Bitmap Heap Scan.
_BAD_SCANS = {"Seq Scan"}
_GOOD_SCANS = {"Index Scan", "Index Only Scan", "Bitmap Heap Scan", "Bitmap Index Scan"}


def _row_estimate_ratio(node: dict[str, Any]) -> float | None:
    """How many times off was the planner's row estimate?"""
    plan_rows = node.get("Plan Rows")
    actual_rows = node.get("Actual Rows")
    if plan_rows is None or actual_rows is None:
        return None
    # Symmetric: 100x over and 100x under both look like "100".
    high = max(plan_rows, actual_rows, 1)
    low = max(min(plan_rows, actual_rows), 1)
    return high / low


def _node_self_time_ms(node: dict[str, Any]) -> float:
    """Actual time spent in *this* node (excluding children), in ms.

    Postgres reports "Actual Total Time" as per-loop average, inclusive of
    children, and "Actual Loops" tells us how many times the node ran.
    """
    total = node.get("Actual Total Time")
    loops = node.get("Actual Loops", 1) or 1
    if total is None:
        return 0.0
    inclusive = total * loops
    children_time = 0.0
    for c in node.get("Plans", []) or []:
        c_total = c.get("Actual Total Time")
        c_loops = c.get("Actual Loops", 1) or 1
        if c_total is not None:
            children_time += c_total * c_loops
    return max(0.0, inclusive - children_time)


def _summarize_hot_nodes(plan_root: dict[str, Any], top_n: int = 5) -> list[dict[str, Any]]:
    """Top contributors to runtime, ranked by self-time."""
    rows = []
    for node in walk(plan_root):
        self_ms = _node_self_time_ms(node)
        rows.append({
            "node_type": node.get("Node Type"),
            "relation": node.get("Relation Name") or node.get("Index Name"),
            "self_ms": round(self_ms, 2),
            "actual_rows": node.get("Actual Rows"),
            "plan_rows": node.get("Plan Rows"),
            "loops": node.get("Actual Loops", 1),
        })
    rows.sort(key=lambda r: r["self_ms"], reverse=True)
    return rows[:top_n]


def diff_plans(baseline: dict[str, Any], incident: dict[str, Any],
               baseline_path: str = "baseline", incident_path: str = "incident",
               query: str | None = None) -> DiffReport:
    """Compute the full structural diff between baseline and incident plans."""
    b_root = baseline["Plan"]
    i_root = incident["Plan"]

    report = DiffReport(
        baseline_path=baseline_path,
        incident_path=incident_path,
        baseline_runtime_ms=baseline.get("Execution Time"),
        incident_runtime_ms=incident.get("Execution Time"),
        baseline_planning_ms=baseline.get("Planning Time"),
        incident_planning_ms=incident.get("Planning Time"),
        query=query or baseline.get("Query Text") or incident.get("Query Text"),
        hot_nodes=_summarize_hot_nodes(i_root),
    )

    # --- Top-level runtime delta ------------------------------------------------
    if report.baseline_runtime_ms and report.incident_runtime_ms:
        factor = report.incident_runtime_ms / report.baseline_runtime_ms
        if factor >= 2:
            sev = "critical" if factor >= 10 else "warning"
            report.findings.append(Finding(
                kind="runtime_regression",
                severity=sev,
                summary=(f"Execution time grew {factor:.1f}× "
                         f"({report.baseline_runtime_ms:.0f}ms → {report.incident_runtime_ms:.0f}ms)."),
                detail={
                    "baseline_ms": report.baseline_runtime_ms,
                    "incident_ms": report.incident_runtime_ms,
                    "factor": round(factor, 2),
                },
            ))

    # Scan-method flips on shared relations
    b_scans = {n.get("Relation Name"): n.get("Node Type")
               for n in walk(b_root) if n.get("Relation Name")}
    i_scans = {n.get("Relation Name"): n.get("Node Type")
               for n in walk(i_root) if n.get("Relation Name")}
    for rel, before in b_scans.items():
        after = i_scans.get(rel)
        if not after or before == after:
            continue
        if before in _GOOD_SCANS and after in _BAD_SCANS:
            report.findings.append(Finding(
                kind="scan_method_flip",
                severity="critical",
                summary=f"Table '{rel}' flipped from {before} to {after}.",
                detail={"relation": rel, "before": before, "after": after},
            ))
        elif before != after:
            report.findings.append(Finding(
                kind="scan_method_change",
                severity="info",
                summary=f"Table '{rel}' scan changed: {before} → {after}.",
                detail={"relation": rel, "before": before, "after": after},
            ))

    # Join-method changes (Hash Join ↔ Nested Loop ↔ Merge Join)
    b_joins = [n.get("Node Type") for n in walk(b_root)
               if n.get("Node Type", "").endswith("Join") or n.get("Node Type") == "Nested Loop"]
    i_joins = [n.get("Node Type") for n in walk(i_root)
               if n.get("Node Type", "").endswith("Join") or n.get("Node Type") == "Nested Loop"]
    if sorted(b_joins) != sorted(i_joins):
        report.findings.append(Finding(
            kind="join_strategy_change",
            severity="warning",
            summary=f"Join strategies changed: {b_joins} → {i_joins}.",
            detail={"before": b_joins, "after": i_joins},
        ))

    # Row-estimate miss on the incident plan (anywhere ≥ 100×)
    for node in walk(i_root):
        ratio = _row_estimate_ratio(node)
        if ratio is None or ratio < 100:
            continue
        sev = "critical" if ratio >= 1000 else "warning"
        report.findings.append(Finding(
            kind="row_estimate_miss",
            severity=sev,
            summary=(f"{node.get('Node Type')} on "
                     f"{node.get('Relation Name') or node.get('Index Name') or '?'} "
                     f"misestimated rows by {ratio:.0f}× "
                     f"(planned {node.get('Plan Rows')}, actual {node.get('Actual Rows')})."),
            detail={
                "node_type": node.get("Node Type"),
                "relation": node.get("Relation Name") or node.get("Index Name"),
                "plan_rows": node.get("Plan Rows"),
                "actual_rows": node.get("Actual Rows"),
                "ratio": round(ratio, 1),
            },
        ))

    # Sort spilled to disk?
    for node in walk(i_root):
        if node.get("Node Type") != "Sort":
            continue
        method = node.get("Sort Method", "")
        if "Disk" in method:
            report.findings.append(Finding(
                kind="sort_spilled_to_disk",
                severity="warning",
                summary=f"Sort spilled to disk ({method}, {node.get('Sort Space Used', '?')} kB).",
                detail={
                    "sort_method": method,
                    "space_kb": node.get("Sort Space Used"),
                },
            ))

    # Hash batches > 1 means hash table didn't fit in work_mem
    for node in walk(i_root):
        if node.get("Node Type") != "Hash":
            continue
        batches = node.get("Hash Batches", 1) or 1
        orig = node.get("Original Hash Batches", batches) or batches
        if batches > 1:
            report.findings.append(Finding(
                kind="hash_spilled_to_disk",
                severity="warning",
                summary=(f"Hash spilled into {batches} batches "
                         f"(planned {orig}); work_mem is too small for this build."),
                detail={"hash_batches": batches, "original_hash_batches": orig},
            ))

    # Parallel-worker regression
    b_workers = sum((n.get("Workers Launched") or 0) for n in walk(b_root))
    i_workers = sum((n.get("Workers Launched") or 0) for n in walk(i_root))
    if b_workers and not i_workers:
        report.findings.append(Finding(
            kind="parallelism_lost",
            severity="warning",
            summary=f"Lost parallelism: baseline used {b_workers} workers, incident used 0.",
            detail={"baseline_workers": b_workers, "incident_workers": 0},
        ))

    # Nested Loop with very large outer side — the classic parameter-sniffing footprint
    for node in walk(i_root):
        if node.get("Node Type") != "Nested Loop":
            continue
        outer = (node.get("Plans") or [None])[0]
        if not outer:
            continue
        outer_rows = outer.get("Actual Rows") or 0
        if outer_rows >= 10_000:
            report.findings.append(Finding(
                kind="nested_loop_blowup",
                severity="critical",
                summary=(f"Nested Loop with {outer_rows:,} outer rows, "
                         "each one triggers an inner scan; the join strategy is likely wrong."),
                detail={"outer_rows": outer_rows},
            ))

    return report
