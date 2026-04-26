"""
FastAPI app for Global Logistics Resolver (OpenEnv-compatible HTTP API).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root on path for env / models / tasks at project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from env import LogisticsEnv
from models import LogisticsAction, LogisticsReward
from tasks import LogisticsGrader, open_unit_score

app = FastAPI(title="OpenEnv - Global Supply Chain & Logistics Resolver")

env = LogisticsEnv()
current_task_level = "easy"


_ROOT_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Global Logistics Resolver · OpenEnv API</title>
  <style>
    :root {
      --bg0: #0f172a;
      --bg1: #1e1b4b;
      --card: rgba(255, 255, 255, 0.06);
      --card-border: rgba(255, 255, 255, 0.12);
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #818cf8;
      --accent2: #38bdf8;
      --code-bg: #020617;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      color: var(--text);
      background: radial-gradient(1200px 600px at 80% -10%, rgba(99, 102, 241, 0.35), transparent),
                  radial-gradient(900px 500px at -10% 100%, rgba(14, 165, 233, 0.2), transparent),
                  linear-gradient(165deg, var(--bg0) 0%, var(--bg1) 45%, #0f172a 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
    }
    .shell { width: 100%; max-width: 32rem; }
    .card {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 1rem;
      padding: 1.75rem 1.75rem 1.5rem;
      backdrop-filter: blur(12px);
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.45);
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--accent2);
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid rgba(56, 189, 248, 0.25);
      padding: 0.25rem 0.6rem;
      border-radius: 999px;
      margin-bottom: 1rem;
    }
    .badge-dot {
      width: 6px; height: 6px;
      border-radius: 50%;
      background: #34d399;
      box-shadow: 0 0 8px #34d399;
    }
    h1 {
      margin: 0 0 0.35rem;
      font-size: 1.45rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      line-height: 1.2;
    }
    .sub {
      margin: 0 0 1.35rem;
      font-size: 0.9rem;
      color: var(--muted);
      line-height: 1.55;
    }
    .links { display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 1.25rem; }
    .links a {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      padding: 0.65rem 0.85rem;
      border-radius: 0.6rem;
      text-decoration: none;
      color: var(--text);
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      font-size: 0.875rem;
      font-weight: 500;
      transition: background 0.15s ease, border-color 0.15s ease, transform 0.15s ease;
    }
    .links a:hover {
      background: rgba(129, 140, 248, 0.15);
      border-color: rgba(129, 140, 248, 0.35);
      transform: translateY(-1px);
    }
    .links a span.path { font-family: ui-monospace, monospace; font-size: 0.75rem; color: var(--muted); }
    .section-label {
      font-size: 0.65rem;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 0.4rem;
    }
    pre#health {
      margin: 0;
      padding: 0.75rem 1rem;
      border-radius: 0.5rem;
      background: var(--code-bg);
      border: 1px solid rgba(148, 163, 184, 0.15);
      font-family: ui-monospace, "Cascadia Code", monospace;
      font-size: 0.78rem;
      line-height: 1.45;
      color: #e2e8f0;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-all;
    }
    footer {
      margin-top: 1.1rem;
      padding-top: 1rem;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
      font-size: 0.75rem;
      color: var(--muted);
    }
    footer a {
      color: var(--accent);
      text-decoration: none;
    }
    footer a:hover { text-decoration: underline; }
    @media (prefers-reduced-motion: reduce) {
      .links a { transition: none; }
      .links a:hover { transform: none; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="card">
      <div class="badge"><span class="badge-dot"></span> Service online</div>
      <h1>🚛 Global Logistics Resolver</h1>
      <p class="sub">OpenEnv-compatible FastAPI — reset, step, and grade a multi-truck logistics simulator over HTTP.</p>
      <div class="links">
        <a href="/docs"><span>OpenAPI &amp; try-it console</span><span class="path">/docs</span></a>
        <a href="/health"><span>Health check</span><span class="path">/health</span></a>
        <a href="/tasks"><span>Tasks &amp; action schema</span><span class="path">/tasks</span></a>
      </div>
      <div class="section-label">Live response</div>
      <pre id="health">Loading…</pre>
      <footer>
        JSON root for scripts: <a href="/?format=json">?format=json</a>
        · <code>POST /reset</code> · <code>POST /step</code> · <code>GET /grader</code>
      </footer>
    </div>
  </div>
  <script>
    fetch("/health").then(function (r) { return r.json(); }).then(function (j) {
      document.getElementById("health").textContent = JSON.stringify(j);
    }).catch(function () {
      document.getElementById("health").textContent = '{"error":"Could not reach /health"}';
    });
  </script>
</body>
</html>
"""


@app.get("/")
async def root(request: Request):
    """Space iframe hits `/` — serve a tiny HTML shell; API under `/reset`, `/state`, `/step`, `/docs`."""
    if request.query_params.get("format") == "json":
        return {"status": "ok", "service": "global-logistics-resolver", "docs": "/docs"}
    return HTMLResponse(content=_ROOT_PAGE_HTML)


@app.post("/reset")
async def reset_env(request: Request):
    """Accept empty body, `{}`, or `null` — hackathon graders may POST with no JSON body."""
    global current_task_level
    task_level = "easy"
    try:
        raw = await request.body()
        if raw:
            body = json.loads(raw)
            if isinstance(body, dict) and body.get("task_level") is not None:
                task_level = str(body["task_level"])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    current_task_level = task_level
    env.reset(task_level=current_task_level)
    return {"state": env.public_state()}


@app.get("/state")
def get_state():
    return {"state": env.public_state()}


@app.get("/health")
def health_check():
    return {"status": "healthy", "incident_id": env.state().get("incident_id")}


@app.post("/step")
def step_env(action: LogisticsAction):
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": LogisticsReward(value=reward).model_dump(),
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def get_tasks():
    return {
        "tasks": ["easy", "medium", "hard"],
        "action_schema": LogisticsAction.model_json_schema(),
    }


@app.get("/grader")
def get_grader_score():
    score = LogisticsGrader.evaluate(current_task_level, env.state())
    return {"score": score}


@app.get("/baseline")
def run_baseline():
    # Same raw optimums as before, mapped to (0,1) exclusive via open_unit_score.
    return {
        "baseline_scores": {
            "easy": open_unit_score(1.0),
            "medium": open_unit_score(2 / 3),  # ~$100 remaining on $300 cap scripted optimum
            "hard": open_unit_score(1.0),
        }
    }


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", os.environ.get("SERVER_PORT", "7860")))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("server.app:app", host=host, port=port, factory=False)


if __name__ == "__main__":
    main()
