"""
LLM + logistics env harness.

Hackathon-style variables (OpenAI client):
  API_BASE_URL   OpenAI-compatible LLM base URL (optional; see _resolve_llm_base_url)
  MODEL_NAME     Model id for chat.completions
  HF_TOKEN       Preferred API key name for Hugging Face / router

Also supported:
  OPENAI_API_KEY  Use instead of HF_TOKEN for api.openai.com
  OPENENV_BASE_URL / ENV_BASE_URL  This simulator's HTTP API (not the LLM host)
"""

import os, json, requests, time
from typing import Optional

from openai import OpenAI
from tasks import SCENARIOS

DEFAULT_HF_LLM_BASE_URL = "https://router.huggingface.co/v1"

# Environment server (reset / state / step / grader) — not the LLM host
OPENENV_BASE_URL = (
    os.environ.get("OPENENV_BASE_URL")
    or os.environ.get("ENV_BASE_URL")
    or "http://localhost:7860"
)

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
# Brief lists HF_TOKEN first; OpenAI key is for local / alternate hosts
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(
        "CRITICAL: No API key. Set HF_TOKEN (preferred) or OPENAI_API_KEY."
    )


def _resolve_llm_base_url() -> Optional[str]:
    explicit = os.environ.get("API_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    if os.environ.get("HF_TOKEN"):
        return DEFAULT_HF_LLM_BASE_URL.rstrip("/")
    return None


_llm_base = _resolve_llm_base_url()
_client_kw: dict = {"api_key": API_KEY}
if _llm_base is not None:
    _client_kw["base_url"] = _llm_base
client = OpenAI(**_client_kw)
SCHEMA = """{"action_type": "check_network"|"load_truck"|"route_truck"|"wait", "truck_id": "str?", "warehouse": "str?", "amount": "int?", "route_id": "str?", "hours": "int?"}"""

def run_task(task):
    print(f"\n🚀 MISSION: {task.upper()}")
    requests.post(f"{OPENENV_BASE_URL}/reset", json={"task_level": task})
    brief = SCENARIOS.get(task, "Solve the logistics crisis.")
    
    # System prompt includes the 'Story' and the 'Rules'
    messages = [{"role": "system", "content": f"Brief: {brief}\nSchema: {SCHEMA}\nOutput ONLY raw JSON actions."}]

    for step in range(25):
        state_resp = requests.get(f"{OPENENV_BASE_URL}/state").json()
        state = state_resp.get("state", {})
        
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=messages + [{"role": "user", "content": f"Telemetry: {json.dumps(state)}"}],
                response_format={"type": "json_object"}
            )
            action_json = resp.choices[0].message.content
            action = json.loads(action_json)
            
            # Record the attempt in history
            messages.append({"role": "assistant", "content": action_json})
            
            res = requests.post(f"{OPENENV_BASE_URL}/step", json=action)
            
            if res.status_code != 200:
                # Provide rejection feedback so the model can correct its parameters
                error_feedback = f"SYSTEM_REJECTED: {res.text}. Fix parameters and retry."
                messages.append({"role": "user", "content": error_feedback})
                print(f"Step {step+1} [REJECTED]: {res.text}")
                continue
                
            data = res.json()
            msg = data['observation']['message']
            print(f"Step {step+1}: {msg}")
            
            # Record the successful outcome in history
            messages.append({"role": "user", "content": f"Observation: {msg}"})

            if data["done"]: break
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ Execution Error: {e}"); break
            
    score = requests.get(f"{OPENENV_BASE_URL}/grader").json().get("score", 0.0)
    print(f"🏁 FINAL {task.upper()} SCORE: {score}")
    return score

if __name__ == "__main__":
    results = {}
    for t in ["easy", "medium", "hard"]:
        results[t] = run_task(t)
    print("\n" + "="*30 + "\nBASELINE MISSION SUMMARY\n" + json.dumps(results, indent=2) + "\n" + "="*30)