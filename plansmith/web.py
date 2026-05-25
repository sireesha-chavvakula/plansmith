"""Minimal Flask UI for Plansmith.

Paste two EXPLAIN JSON plans on the left, optionally a SQL string, hit
"Triage", and a Server-Sent-Events stream pours Gemma's report into the right
column as it generates. All processing — including the model — runs locally.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from .plan_diff import diff_plans
from .triage import DEFAULT_MODEL, build_user_prompt, stream_triage


def _load_sample(name: str) -> dict[str, str]:
    """Read a sample baseline/incident/query trio shipped under samples/."""
    root = Path(__file__).resolve().parent.parent / "samples"
    base = (root / f"baseline_{name}.json").read_text() if (root / f"baseline_{name}.json").exists() else ""
    inc  = (root / f"incident_{name}.json").read_text() if (root / f"incident_{name}.json").exists() else ""
    sql  = (root / f"{name}.sql").read_text() if (root / f"{name}.sql").exists() else ""
    return {"baseline": base, "incident": inc, "query": sql}


def create_app(model: str = DEFAULT_MODEL,
               ollama_url: str = "http://localhost:11434") -> Flask:
    app = Flask(__name__)
    app.config.update(MODEL=model, OLLAMA_URL=ollama_url)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            model=model,
            sample_1_json=json.dumps(_load_sample("q1")),
            sample_2_json=json.dumps(_load_sample("q2")),
        )

    @app.post("/api/diff")
    def diff():
        body = request.get_json(force=True)
        try:
            baseline = json.loads(body["baseline"])
            incident = json.loads(body["incident"])
        except (KeyError, json.JSONDecodeError) as e:
            return jsonify({"error": f"could not parse JSON: {e}"}), 400
        if isinstance(baseline, list):
            baseline = baseline[0]
        if isinstance(incident, list):
            incident = incident[0]
        report = diff_plans(baseline, incident, query=body.get("query"))
        return jsonify({
            "summary": {
                "baseline_ms": report.baseline_runtime_ms,
                "incident_ms": report.incident_runtime_ms,
                "slowdown_factor": report.slowdown_factor(),
            },
            "findings": [asdict(f) for f in report.findings],
            "hot_nodes": report.hot_nodes,
            "prompt_preview": build_user_prompt(report),
        })

    @app.post("/api/triage")
    def triage_stream():
        body = request.get_json(force=True)
        baseline = json.loads(body["baseline"])
        incident = json.loads(body["incident"])
        if isinstance(baseline, list):
            baseline = baseline[0]
        if isinstance(incident, list):
            incident = incident[0]
        report = diff_plans(baseline, incident, query=body.get("query"))

        @stream_with_context
        def gen():
            try:
                for piece in stream_triage(report,
                                           model=app.config["MODEL"],
                                           ollama_url=app.config["OLLAMA_URL"]):
                    # SSE: escape newlines so they don't terminate the frame.
                    safe = piece.replace("\r", "").replace("\n", "\\n")
                    yield f"data: {safe}\n\n"
            except Exception as e:  # noqa: BLE001
                yield f"event: error\ndata: {str(e)}\n\n"
            yield "event: done\ndata: end\n\n"

        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    return app
