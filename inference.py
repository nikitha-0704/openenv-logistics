"""
LLM + logistics env harness.

Submission / Phase 2 (LiteLLM proxy — required for automated grading):
  API_BASE_URL   Injected by the platform; OpenAI client base_url (no bypass)
  API_KEY        Injected LiteLLM proxy key (use this, not personal provider keys)
  MODEL_NAME     Often injected; optional default below for local runs

Local development (when API_BASE_URL + API_KEY are not both set):
  API_BASE_URL   Optional default Hugging Face router
  HF_TOKEN or OPENAI_API_KEY

Also:
  OPENENV_BASE_URL / ENV_BASE_URL  Logistics HTTP API (not the LLM host)
  LOCAL_IMAGE_NAME  Only if using from_docker_image() (unused here)

Stdout logs for validators: [START] / [STEP] / [END] with key=value fields.
"""

import json
import os
import time

import requests
from openai import OpenAI
from tasks import SCENARIOS

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Environment server (reset / state / step / grader) — not the LLM host
OPENENV_BASE_URL = (
    os.environ.get("OPENENV_BASE_URL")
    or os.environ.get("ENV_BASE_URL")
    or "http://localhost:7860"
)

# Phase 2: graders inject API_BASE_URL + API_KEY for LiteLLM — must use them when present.
if "API_BASE_URL" in os.environ and "API_KEY" in os.environ:
    API_BASE_URL = os.environ["API_BASE_URL"].rstrip("/")
    MODEL_NAME = os.environ.get("MODEL_NAME", MODEL_NAME)
    client = OpenAI(base_url=API_BASE_URL, api_key=os.environ["API_KEY"])
else:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
    _local_key = HF_TOKEN or os.getenv("OPENAI_API_KEY")
    if not _local_key:
        raise ValueError(
            "Set API_BASE_URL and API_KEY for submission (LiteLLM proxy), or for local runs "
            "set HF_TOKEN or OPENAI_API_KEY."
        )
    client = OpenAI(base_url=API_BASE_URL, api_key=_local_key)

SCHEMA = """{"action_type": "check_network"|"load_truck"|"route_truck"|"wait", "truck_id": "str?", "warehouse": "str?", "amount": "int?", "route_id": "str?", "hours": "int?"}"""


def _kv(*pairs: tuple[str, str]) -> str:
    """Space-separated key=value for validator lines; quote values that need it."""
    out = []
    for k, v in pairs:
        s = str(v)
        if any(c in s for c in ' ="\n\t\r'):
            esc = s.replace("\\", "\\\\").replace('"', '\\"')
            out.append(f'{k}="{esc}"')
        else:
            out.append(f"{k}={s}")
    return " ".join(out)


def _action_compact(action: dict) -> str:
    return json.dumps(action, separators=(",", ":"), ensure_ascii=True)


def run_task(task: str) -> float:
    print(f"[START] {_kv(('task', task), ('env', OPENENV_BASE_URL))}", flush=True)
    requests.post(f"{OPENENV_BASE_URL}/reset", json={"task_level": task})
    brief = SCENARIOS.get(task, "Solve the logistics crisis.")

    messages = [
        {
            "role": "system",
            "content": f"Brief: {brief}\nSchema: {SCHEMA}\nOutput ONLY raw JSON actions.",
        }
    ]

    rewards: list[float] = []
    episode_done = False
    step_ix = 0

    for _ in range(25):
        state_resp = requests.get(f"{OPENENV_BASE_URL}/state").json()
        state = state_resp.get("state", {})
        current_action: dict = {}

        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages
                + [{"role": "user", "content": f"Telemetry: {json.dumps(state)}"}],
                response_format={"type": "json_object"},
            )
            action_json = resp.choices[0].message.content
            current_action = json.loads(action_json)

            messages.append({"role": "assistant", "content": action_json})

            res = requests.post(f"{OPENENV_BASE_URL}/step", json=current_action)

            step_ix += 1

            if res.status_code != 200:
                err = res.text[:2000]
                error_feedback = f"SYSTEM_REJECTED: {res.text}. Fix parameters and retry."
                messages.append({"role": "user", "content": error_feedback})
                print(
                    f"[STEP] {_kv(('step', str(step_ix)), ('reward', '0'), ('action', _action_compact(current_action)), ('error', err))}",
                    flush=True,
                )
                continue

            data = res.json()
            msg = data["observation"]["message"]
            done = bool(data.get("done"))
            rw = data.get("reward", {})
            reward_val = float(rw.get("value", 0.0))
            rewards.append(reward_val)

            print(
                f"[STEP] {_kv(('step', str(step_ix)), ('reward', str(reward_val)), ('action', _action_compact(current_action)), ('error', ''))}",
                flush=True,
            )

            messages.append({"role": "user", "content": f"Observation: {msg}"})

            if done:
                episode_done = True
                break
            time.sleep(0.5)
        except Exception as e:
            step_ix += 1
            act_s = _action_compact(current_action) if current_action else "{}"
            print(
                f"[STEP] {_kv(('step', str(step_ix)), ('reward', '0'), ('action', act_s), ('error', str(e)))}",
                flush=True,
            )
            break

    score = requests.get(f"{OPENENV_BASE_URL}/grader").json().get("score", 0.0)
    rewards_lit = json.dumps(rewards, separators=(",", ":"))
    print(
        f"[END] {_kv(('task', task), ('success', str(episode_done).lower()), ('steps', str(step_ix)), ('rewards', rewards_lit), ('score', str(score)))}",
        flush=True,
    )
    return float(score)


if __name__ == "__main__":
    results = {}
    for t in ["easy", "medium", "hard"]:
        results[t] = run_task(t)
    print(json.dumps({"baseline_mission_summary": results}, indent=2), flush=True)
