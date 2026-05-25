# Plansmith

**Local-first Postgres slow-query triage with Gemma 4 E4B.**

Drop in a baseline `EXPLAIN (ANALYZE, FORMAT JSON)` plan from when the query
was fast, and the incident plan from now. Plansmith does a deterministic
structural diff (scan-method flips, row-estimate misses, sort and hash
spills, parallelism loss, nested-loop blowups) and then asks a local Gemma 4
E4B model to turn that into a triage runbook with ranked root causes,
immediate mitigations, and permanent fixes.

**Everything runs on your machine.** Query plans contain schema, business
logic, and sometimes literal data values. Sending them to a cloud model is a
non-starter at most companies. Plansmith never opens an outbound connection
beyond `localhost:11434`, which is your Ollama daemon.

## Why Gemma 4 E4B specifically

| Need                                              | Why E4B fits                                                          |
| ------------------------------------------------- | --------------------------------------------------------------------- |
| Plans must not leave the host                     | Runs locally via Ollama, no cloud round-trip                          |
| Plans can be very large                           | 128K context absorbs even the chunkiest plans without truncation      |
| 3 AM triage needs fast first-token latency        | Q4_K_M at about 8 GB fits in laptop RAM and streams on CPU or Metal   |
| You don't want the model inventing numbers        | A 4B-class model does fine when given a pre-chewed structured diff    |
| You may want to deploy on a sidecar or a Pi       | Same model tag works from edge to dev box                             |

The design choice that makes this work: **the LLM never reads raw JSON
plans.** A deterministic Python pass extracts the diff first and hands the
model a compact JSON of *named findings* with measured numbers. The model's
job is narrower (explanation, ranking, runbook prose) which is exactly what
a 4B edge model is good at.

## Quick start

You need [Ollama](https://ollama.com) running locally with `gemma4:e4b`
pulled:

```bash
ollama pull gemma4:e4b
ollama serve   # (usually already running on macOS)
```

Then install Plansmith into a venv:

```bash
cd plansmith
python3 -m venv .venv
.venv/bin/pip install -e .
```

Triage a sample regression (parameter-sniffing, 497× slower):

```bash
.venv/bin/plansmith analyze \
    --baseline samples/baseline_q1.json \
    --incident samples/incident_q1.json \
    --query    samples/q1.sql
```

A second sample showing stats drift + hash spill (92× slower):

```bash
.venv/bin/plansmith analyze \
    --baseline samples/baseline_q2.json \
    --incident samples/incident_q2.json \
    --query    samples/q2.sql
```

Or open the web UI:

```bash
.venv/bin/plansmith serve
# → http://127.0.0.1:8765
```

Paste two `EXPLAIN (ANALYZE, FORMAT JSON)` outputs, hit **Triage**, and the
runbook streams in from your local Gemma 4. The page also has a theme
switcher (system, light, dark) in the top right.

## What the diff catches (without the LLM)

Even with `--no-llm`, Plansmith is useful as a fast structural diff:

| Finding              | What it means                                                  |
| -------------------- | -------------------------------------------------------------- |
| `runtime_regression` | top-line slowdown ratio                                        |
| `scan_method_flip`   | Index Scan → Seq Scan on the same relation                     |
| `row_estimate_miss`  | planner's row count off by ≥ 100×                              |
| `nested_loop_blowup` | Nested Loop with a huge outer side (classic param-sniffing)    |
| `sort_spilled_to_disk` | `work_mem` too small for the sort                            |
| `hash_spilled_to_disk` | Hash Build couldn't fit, batches > 1                         |
| `parallelism_lost`   | baseline used workers, incident dropped to 0                   |
| `join_strategy_change` | Hash Join ↔ Nested Loop ↔ Merge Join                         |

The structured findings are also exposed in JSON via `POST /api/diff` so you
can wire Plansmith into existing observability pipelines.

## How to capture an EXPLAIN plan to feed in

```sql
EXPLAIN (ANALYZE, FORMAT JSON, BUFFERS, VERBOSE)
SELECT ...;   -- your slow query
```

Save the output to a file. The same goes for the baseline, keep one
"healthy" plan per important query under version control.

## Project layout

```
plansmith/
├── plansmith/
│   ├── plan_diff.py     # deterministic EXPLAIN JSON diff & findings
│   ├── triage.py        # Ollama / Gemma 4 prompt + streaming
│   ├── cli.py           # `plansmith analyze` / `plansmith serve`
│   ├── web.py           # Flask app (SSE streaming UI)
│   └── templates/index.html
├── samples/
│   ├── q1.sql           # scenario 1: parameter sniffing
│   ├── baseline_q1.json
│   ├── incident_q1.json
│   ├── q2.sql           # scenario 2: stats drift + hash spill
│   ├── baseline_q2.json
│   └── incident_q2.json
├── pyproject.toml
├── LICENSE              # Apache-2.0
└── README.md
```

## License

Apache-2.0. See `LICENSE`.

---

Built for the [Gemma 4 Challenge](https://dev.to/t/gemmachallenge), May 2026.
