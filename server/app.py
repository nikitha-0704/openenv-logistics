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
  <title>Global Logistics Resolver API</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 40rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
    a { color: #2563eb; }
    pre { background: #f4f4f5; padding: 0.75rem 1rem; border-radius: 6px; overflow-x: auto; font-size: 0.9rem; }
    .muted { color: #71717a; font-size: 0.875rem; }
  </style>
</head>
<body>
  <h1>Global Logistics Resolver API</h1>
  <p>OpenEnv-compatible HTTP API. Interactive routes:</p>
  <ul>
    <li><a href="/docs"><strong>OpenAPI docs</strong> (/docs)</a></li>
    <li><a href="/health">Health JSON</a> (/health)</li>
    <li><a href="/tasks">Task list + action schema</a> (/tasks)</li>
  </ul>
  <p class="muted">Live health (same payload as <code>/health</code>):</p>
  <pre id="health">Loading…</pre>
  <p class="muted">Machine-readable root: <a href="/?format=json"><code>?format=json</code></a></p>
  <script>
    fetch("/health").then(function (r) { return r.json(); }).then(function (j) {
      document.getElementById("health").textContent = JSON.stringify(j);
    }).catch(function () {
      document.getElementById("health").textContent = "(Could not load /health — open the link above.)";
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
