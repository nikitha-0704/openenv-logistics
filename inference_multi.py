"""Hierarchical multi-agent harness (Dispatcher + per-truck Drivers + message bus).

Same env API and the same `[START]/[STEP]/[END]` log shape as `inference.py`,
plus extra log lines so reviewers can see coordination unfold:

  [PLAN]   …                         initial Dispatcher plan
  [MSG]    from=Tnnn body="…"        a Driver's broadcast on the message bus
  [REPLAN] reason="…" plan=…         Dispatcher re-plan after a [BREAKING] event
  [STEP]   step=N agent=Tnnn …       a Driver's env action and reward

`tag=multi` is on `[START]` and `[END]` so the plot script can split runs.

Run:
    python inference_multi.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from agents import Dispatcher, Driver, MessageBus
from agents.driver import split_broadcast
from agents.llm import OPENENV_BASE_URL
from tasks import SCENARIOS

_EPISODE_MEMORY_PATH = Path(__file__).resolve().parent / ".episode_memory.txt"
_EPISODE_MEMORY_MAX_CHARS = 2000

# Cap total env steps per episode so a misbehaving Driver can't loop forever.
MAX_STEPS_PER_EPISODE = int(os.getenv("MULTI_MAX_STEPS", "40"))
# How many recent peer messages to surface to each Driver.
PEER_MSG_LIMIT = int(os.getenv("MULTI_PEER_MSGS", "4"))


def _load_episode_memory() -> str:
    try:
        if not _EPISODE_MEMORY_PATH.is_file():
            return ""
        raw = _EPISODE_MEMORY_PATH.read_text(encoding="utf-8", errors="replace")
        return raw[-_EPISODE_MEMORY_MAX_CHARS:] if len(raw) > _EPISODE_MEMORY_MAX_CHARS else raw
    except OSError:
        return ""


def _append_episode_memory(line: str) -> None:
    try:
        with open(_EPISODE_MEMORY_PATH, "a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
    except OSError:
        pass


def _kv(*pairs: tuple[str, str]) -> str:
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


def _truck_ids(state: dict) -> list[str]:
    return list((state.get("trucks") or {}).keys())


def _assignments_by_truck(plan: dict) -> dict[str, dict]:
    return {a["truck_id"]: a for a in plan.get("assignments", []) if a.get("truck_id")}


def run_task(task: str) -> float:
    print(f"[START] {_kv(('task', task), ('env', OPENENV_BASE_URL), ('tag', 'multi'))}", flush=True)

    reset_resp = requests.post(f"{OPENENV_BASE_URL}/reset", json={"task_level": task}).json()
    state0 = reset_resp.get("state") or {}
    brief = state0.get("mission_brief") or SCENARIOS.get(task, "Solve the logistics crisis.")
    constraints = state0.get("instruction_constraints") or []
    memory_tail = _load_episode_memory()

    dispatcher = Dispatcher()
    plan = dispatcher.plan(
        mission_brief=brief,
        instruction_constraints=constraints,
        public_state=state0,
        memory_tail=memory_tail,
    )

    truck_ids = _truck_ids(state0) or [a["truck_id"] for a in plan["assignments"]]
    assignments = _assignments_by_truck(plan)

    drivers: dict[str, Driver] = {}
    for tid in truck_ids:
        d = Driver(truck_id=tid)
        d.reset(assignments.get(tid, {"truck_id": tid}), brief)
        drivers[tid] = d

    bus = MessageBus()

    print(
        f"[PLAN] {_kv(('task', task), ('telemetry_first', str(plan.get('telemetry_first', True)).lower()), ('plan', json.dumps(plan, separators=(',', ':'))))}",
        flush=True,
    )

    rewards: list[float] = []
    episode_done = False
    step_ix = 0
    replan_count = 0

    if plan.get("telemetry_first", True):
        res = requests.post(
            f"{OPENENV_BASE_URL}/step", json={"action_type": "check_network"}
        )
        step_ix += 1
        if res.status_code == 200:
            data = res.json()
            rw = float((data.get("reward") or {}).get("value", 0.0))
            rewards.append(rw)
            print(
                f"[STEP] {_kv(('step', str(step_ix)), ('agent', 'dispatcher'), ('reward', str(rw)), ('action', '{\"action_type\":\"check_network\"}'), ('error', ''))}",
                flush=True,
            )
            for d in drivers.values():
                d.record_observation(data["observation"]["message"])
            if data.get("done"):
                episode_done = True

    turn = 0
    while not episode_done and step_ix < MAX_STEPS_PER_EPISODE:
        public = requests.get(f"{OPENENV_BASE_URL}/state").json().get("state", {})
        tid = truck_ids[turn % len(truck_ids)]
        turn += 1
        driver = drivers[tid]

        try:
            raw_action = driver.next_action(
                public, peer_messages=bus.recent(exclude_sender=tid, limit=PEER_MSG_LIMIT)
            )
        except Exception as exc:  # noqa: BLE001 - log and skip a turn
            step_ix += 1
            print(
                f"[STEP] {_kv(('step', str(step_ix)), ('agent', tid), ('reward', '0'), ('action', '{}'), ('error', str(exc)))}",
                flush=True,
            )
            continue

        env_action, broadcast = split_broadcast(dict(raw_action))
        if broadcast:
            bus.post(tid, broadcast)
            print(
                f"[MSG] {_kv(('from', tid), ('body', broadcast))}",
                flush=True,
            )

        try:
            res = requests.post(f"{OPENENV_BASE_URL}/step", json=env_action)
        except requests.RequestException as exc:
            step_ix += 1
            print(
                f"[STEP] {_kv(('step', str(step_ix)), ('agent', tid), ('reward', '0'), ('action', _action_compact(env_action)), ('error', str(exc)))}",
                flush=True,
            )
            break

        step_ix += 1
        if res.status_code != 200:
            err = res.text[:2000]
            driver.record_observation(f"SYSTEM_REJECTED: {err}. Fix parameters and retry.")
            print(
                f"[STEP] {_kv(('step', str(step_ix)), ('agent', tid), ('reward', '0'), ('action', _action_compact(env_action)), ('error', err))}",
                flush=True,
            )
            continue

        data = res.json()
        msg = data["observation"]["message"]
        done = bool(data.get("done"))
        rw = float((data.get("reward") or {}).get("value", 0.0))
        rewards.append(rw)
        driver.record_observation(msg)

        print(
            f"[STEP] {_kv(('step', str(step_ix)), ('agent', tid), ('reward', str(rw)), ('action', _action_compact(env_action)), ('error', ''))}",
            flush=True,
        )

        # Re-plan trigger: Dispatcher reacts once to the first BREAKING event.
        if "[BREAKING]" in msg and replan_count == 0:
            replan_count += 1
            new_public = requests.get(f"{OPENENV_BASE_URL}/state").json().get("state", {})
            try:
                new_plan = dispatcher.plan(
                    mission_brief=brief,
                    instruction_constraints=constraints,
                    public_state=new_public,
                    memory_tail=memory_tail,
                    prior_plan=plan,
                    replan_reason=msg,
                )
            except Exception as exc:  # noqa: BLE001
                new_plan = plan
                print(f"[REPLAN] {_kv(('error', str(exc)))}", flush=True)
            else:
                plan = new_plan
                assignments = _assignments_by_truck(new_plan)
                for sub_tid, drv in drivers.items():
                    drv.reassign(
                        assignments.get(sub_tid, {"truck_id": sub_tid}),
                        brief,
                        reason=msg,
                    )
                print(
                    f"[REPLAN] {_kv(('reason', msg), ('plan', json.dumps(new_plan, separators=(',', ':'))))}",
                    flush=True,
                )

        if done:
            episode_done = True
            break
        time.sleep(0.25)

    score = float(requests.get(f"{OPENENV_BASE_URL}/grader").json().get("score", 0.0))
    rewards_lit = json.dumps(rewards, separators=(",", ":"))
    print(
        f"[END] {_kv(('task', task), ('tag', 'multi'), ('success', str(episode_done).lower()), ('steps', str(step_ix)), ('replans', str(replan_count)), ('rewards', rewards_lit), ('score', str(score)))}",
        flush=True,
    )
    _append_episode_memory(
        f"tag=multi task={task} success={episode_done} steps={step_ix} replans={replan_count} score={score} rewards={rewards_lit}"
    )
    return score


if __name__ == "__main__":
    results = {}
    for t in ["easy", "medium", "hard"]:
        results[t] = run_task(t)
    print(json.dumps({"multi_agent_summary": results}, indent=2), flush=True)
