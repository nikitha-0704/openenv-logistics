"""Driver agent: executes a per-truck assignment via the OpenEnv HTTP API.

Each Driver decides ONE next action (single JSON object matching the env schema)
given (a) its assignment, (b) the latest public/sync state, and (c) its own
short message history. It does NOT plan globally — that is the Dispatcher's job.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from agents.llm import MODEL_NAME, client

ACTION_SCHEMA = (
    '{"action_type":"check_network"|"load_truck"|"route_truck"|"wait",'
    '"truck_id":"str?","warehouse":"str?","amount":"int?",'
    '"route_id":"str?","hours":"int?","broadcast":"str?"}'
)

_VALID_ACTION_KEYS = {
    "action_type",
    "truck_id",
    "warehouse",
    "amount",
    "route_id",
    "hours",
}


def split_broadcast(action: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Pop the side-channel `broadcast` field off a Driver's JSON output.

    Returns `(env_action, broadcast)` where `env_action` only contains keys
    accepted by the env's `LogisticsAction` model.
    """
    raw_broadcast = action.pop("broadcast", "")
    broadcast = str(raw_broadcast).strip() if raw_broadcast else ""
    env_action = {k: v for k, v in action.items() if k in _VALID_ACTION_KEYS}
    return env_action, broadcast


class Driver:
    """Per-truck executor. Owns its own short message history.

    Drivers may include a `broadcast` field on their JSON output; the orchestrator
    strips it before posting to `/step` and replays the most recent peer messages
    into the next prompt via `next_action(..., peer_messages=...)`.
    """

    def __init__(self, truck_id: str, model_name: Optional[str] = None) -> None:
        self.truck_id = truck_id
        self.model_name = model_name or MODEL_NAME
        self._history: list[dict[str, str]] = []
        self._assignment: dict[str, Any] = {}

    def _build_system(self, mission_brief: str) -> str:
        return (
            f"You are Driver {self.truck_id} in a multi-agent ops team. You execute ONE "
            "action per turn. The Dispatcher decomposed the mission; follow your assignment "
            "unless telemetry contradicts it. Output strict JSON only.\n"
            f"Assignment: {json.dumps(self._assignment)}\n"
            f"Mission brief: {mission_brief}\n"
            f"Schema: {ACTION_SCHEMA}\n"
            "If the assignment requires moving on a route whose status is `unknown`, run "
            "check_network first. Never dispatch on a CLOSED route.\n"
            "You MAY include a `broadcast` string (<= 200 chars) to coordinate with peers — "
            "use it to claim an order, surface a blocker, or hand off."
        )

    def reset(self, assignment: dict[str, Any], mission_brief: str) -> None:
        self._assignment = dict(assignment or {})
        self._history = [{"role": "system", "content": self._build_system(mission_brief)}]

    def reassign(self, assignment: dict[str, Any], mission_brief: str, reason: str = "") -> None:
        """Swap in a fresh Dispatcher assignment without losing prior context.

        Adds an updated system message at the head and a user-visible note about
        the re-plan reason (typically a `[BREAKING]` event).
        """
        self._assignment = dict(assignment or {})
        new_system = self._build_system(mission_brief)
        if self._history and self._history[0]["role"] == "system":
            self._history[0] = {"role": "system", "content": new_system}
        else:
            self._history.insert(0, {"role": "system", "content": new_system})
        if reason:
            self._history.append(
                {"role": "user", "content": f"[REPLAN] New assignment in effect: {reason}"}
            )

    def next_action(
        self,
        public_state: dict[str, Any],
        peer_messages: Optional[list[tuple[str, str]]] = None,
    ) -> dict[str, Any]:
        truck = (public_state.get("trucks") or {}).get(self.truck_id, {})
        prompt_state = {
            "telemetry_visibility": public_state.get("telemetry_visibility"),
            "my_truck": truck,
            "warehouses": public_state.get("warehouses"),
            "routes": public_state.get("routes"),
            "active_orders": public_state.get("active_orders"),
            "budget": public_state.get("budget"),
            "current_time_hours": public_state.get("current_time_hours"),
            "network_sync_count": public_state.get("network_sync_count"),
        }
        peer_block = ""
        if peer_messages:
            lines = "\n".join(f"- {sender}: {body}" for sender, body in peer_messages)
            peer_block = f"\nRecent peer messages:\n{lines}"
        user_content = f"Telemetry: {json.dumps(prompt_state)}{peer_block}"
        msgs = self._history + [{"role": "user", "content": user_content}]
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=msgs,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            action = {"action_type": "wait", "hours": 1}
        if not isinstance(action, dict):
            action = {"action_type": "wait", "hours": 1}
        if action.get("action_type") in {"load_truck", "route_truck"}:
            action.setdefault("truck_id", self.truck_id)
        self._history.append({"role": "assistant", "content": raw})
        return action

    def record_observation(self, message: str) -> None:
        self._history.append({"role": "user", "content": f"Observation: {message}"})
