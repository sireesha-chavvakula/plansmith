"""Talk to a local Ollama instance running gemma4:e4b.

We deliberately keep the model's job narrow: it does *not* do arithmetic on
raw JSON. The deterministic diff has already extracted the numbers and named
the patterns; Gemma's contribution is judgment, explanation, and a runbook.
"""

from __future__ import annotations

import json
from typing import Iterator

import requests

from .plan_diff import DiffReport


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:e4b"


SYSTEM_PROMPT = """You are Plansmith, a senior Postgres performance engineer.

You are paged to triage a slow query. A deterministic analyzer has ALREADY
diffed the baseline plan against the incident plan and given you a structured
findings list with measured numbers — trust those numbers; do not invent new
ones, and do not contradict them.

Your job is to produce a triage report in this exact Markdown structure:

## TL;DR
One sentence: what changed and what's the likely cause.

## What the plans say
2–4 short bullets summarizing the structural changes in plain English.
Reference the named findings; do not restate raw row counts the analyzer
already gave.

## Likely root causes (ranked)
A numbered list of 2–4 hypotheses, most likely first. For each:
- one-line hypothesis
- one-line evidence ("supported by finding X")

## Immediate mitigations (minutes)
2–4 things an on-call engineer can do RIGHT NOW. Concrete SQL or settings.
Prefer reversible changes (SET LOCAL, ANALYZE, plan hints) over schema changes.

## Permanent fixes (days)
2–4 durable fixes — index changes, query rewrites, statistics targets,
extended statistics, partitioning.

## Verification
One sentence: how to confirm the fix worked.

Rules:
- Be terse. No filler, no apologies, no "I think". Bullets over paragraphs.
- If a finding is missing data, say so; do not guess.
- Never claim to have run anything yourself.
"""


def build_user_prompt(report: DiffReport) -> str:
    """Pack the structured diff into a compact, model-friendly prompt."""
    payload = {
        "baseline_runtime_ms": report.baseline_runtime_ms,
        "incident_runtime_ms": report.incident_runtime_ms,
        "slowdown_factor": report.slowdown_factor(),
        "baseline_planning_ms": report.baseline_planning_ms,
        "incident_planning_ms": report.incident_planning_ms,
        "findings": [
            {
                "kind": f.kind,
                "severity": f.severity,
                "summary": f.summary,
                "detail": f.detail,
            }
            for f in report.findings
        ],
        "hot_nodes_in_incident": report.hot_nodes,
    }
    pieces = [
        "STRUCTURED DIFF (authoritative — these numbers are measured):",
        "```json",
        json.dumps(payload, indent=2, default=str),
        "```",
    ]
    if report.query:
        pieces += ["", "QUERY TEXT:", "```sql", report.query.strip(), "```"]
    pieces += ["", "Produce the triage report now."]
    return "\n".join(pieces)


def stream_triage(report: DiffReport, *, model: str = DEFAULT_MODEL,
                  ollama_url: str = DEFAULT_OLLAMA_URL,
                  temperature: float = 0.2) -> Iterator[str]:
    """Stream the model's triage report token-by-token.

    Uses Ollama's /api/chat with stream=true. Yields response chunks as they
    arrive so the CLI/web UI can show progress during a 3am incident.
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(report)},
        ],
        "stream": True,
        "options": {"temperature": temperature},
    }
    with requests.post(f"{ollama_url}/api/chat", json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = chunk.get("message", {})
            piece = msg.get("content")
            if piece:
                yield piece
            if chunk.get("done"):
                break


def triage(report: DiffReport, **kwargs) -> str:
    """Non-streaming convenience wrapper — returns the full Markdown report."""
    return "".join(stream_triage(report, **kwargs))
