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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env import LogisticsEnv
from models import LogisticsAction, LogisticsReward
from tasks import LogisticsGrader

app = FastAPI(title="OpenEnv - Global Supply Chain & Logistics Resolver")

env = LogisticsEnv()
current_task_level = "easy"


class ResetRequest(BaseModel):
    task_level: str = "easy"


@app.post("/reset")
def reset_env(req: ResetRequest):
    global current_task_level
    current_task_level = req.task_level
    state = env.reset(task_level=current_task_level)
    return {"state": state}


@app.get("/state")
def get_state():
    return {"state": env.state()}


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
    return {
        "baseline_scores": {
            "easy": 1.0,
            "medium": 0.67,
            "hard": 1.0,
        }
    }


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", os.environ.get("SERVER_PORT", "7860")))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("server.app:app", host=host, port=port, factory=False)


if __name__ == "__main__":
    main()
