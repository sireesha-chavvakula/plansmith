"""Plansmith CLI entry point.

Usage:
    plansmith analyze --baseline samples/baseline_q1.json \\
                      --incident samples/incident_q1.json \\
                      [--query samples/q1.sql] \\
                      [--model gemma4:e4b] \\
                      [--no-llm]
    plansmith serve [--port 8765]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .plan_diff import diff_plans, load_plan
from .triage import DEFAULT_MODEL, stream_triage


console = Console()

_SEVERITY_STYLE = {"critical": "bold red", "warning": "yellow", "info": "cyan"}


def _render_findings_table(report) -> Table:
    t = Table(title="Structured diff findings", show_lines=False, expand=True)
    t.add_column("Severity", style="bold")
    t.add_column("Kind")
    t.add_column("Summary")
    for f in report.findings:
        style = _SEVERITY_STYLE.get(f.severity, "white")
        t.add_row(f"[{style}]{f.severity.upper()}[/{style}]", f.kind, f.summary)
    return t


def cmd_analyze(args: argparse.Namespace) -> int:
    baseline = load_plan(args.baseline)
    incident = load_plan(args.incident)
    query_text = Path(args.query).read_text() if args.query else None
    report = diff_plans(baseline, incident,
                        baseline_path=args.baseline,
                        incident_path=args.incident,
                        query=query_text)

    header_bits = []
    if report.baseline_runtime_ms and report.incident_runtime_ms:
        header_bits.append(f"baseline {report.baseline_runtime_ms:.0f}ms")
        header_bits.append(f"incident {report.incident_runtime_ms:.0f}ms")
        factor = report.slowdown_factor()
        if factor:
            header_bits.append(f"{factor:.1f}× slower")
    console.print(Panel.fit(" · ".join(header_bits) or "(no runtime info)",
                            title="Plansmith", border_style="cyan"))
    console.print(_render_findings_table(report))

    if args.no_llm:
        return 0

    console.print()
    console.rule(f"[bold magenta]Triage report from {args.model}")
    console.print()
    try:
        # Stream raw to stdout so it feels responsive, then re-render as markdown.
        buf: list[str] = []
        for piece in stream_triage(report, model=args.model, ollama_url=args.ollama_url):
            sys.stdout.write(piece)
            sys.stdout.flush()
            buf.append(piece)
    except Exception as e:  # noqa: BLE001 — surface anything from Ollama clearly
        console.print(f"\n[bold red]Ollama call failed:[/bold red] {e}")
        console.print(f"[dim]Is `ollama serve` running and is '{args.model}' pulled?[/dim]")
        return 2
    console.print()
    console.rule()
    console.print(Markdown("".join(buf)))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .web import create_app
    app = create_app(model=args.model, ollama_url=args.ollama_url)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="plansmith",
                                description="Local-first Postgres slow-query triage with Gemma 4 E4B")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Ollama model tag (default: {DEFAULT_MODEL})")
    p.add_argument("--ollama-url", default="http://localhost:11434")

    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="Triage a baseline-vs-incident plan pair")
    a.add_argument("--baseline", required=True, help="Path to baseline EXPLAIN JSON")
    a.add_argument("--incident", required=True, help="Path to incident EXPLAIN JSON")
    a.add_argument("--query", default=None, help="Optional path to the SQL text")
    a.add_argument("--no-llm", action="store_true",
                   help="Skip the LLM step; print only the deterministic diff")
    a.set_defaults(func=cmd_analyze)

    s = sub.add_parser("serve", help="Run the small web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
