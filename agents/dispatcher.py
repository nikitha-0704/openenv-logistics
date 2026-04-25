"""Dispatcher agent: assigns orders to trucks under partial telemetry.

Outputs a strict JSON `Plan`:

    {
      "rationale": "<short string>",
      "assignments": [
        {"truck_id": "T101", "order_id": "ORD-EASY-01", "priority": 1,
         "route_hint": ["North_to_South", "South_to_East"], "load_amount": 20}
      ],
      "telemetry_first": true
    }

The per-truck Driver agents read these assignments and act on them.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from agents.llm import MODEL_NAME, client

DISPATCHER_SCHEMA = (
    '{"rationale":"str","telemetry_first":bool,'
    '"assignments":[{"truck_id":"str","order_id":"str|null","priority":1,'
    '"route_hint":["str"],"load_amount":int}]}'
)


class Dispatcher:
    """Hierarchical planner that turns a mission brief into per-truck assignments."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or MODEL_NAME

    def plan(
        self,
        *,
        mission_brief: str,
        instruction_constraints: list[str],
        public_state: dict[str, Any],
        memory_tail: str = "",
        prior_plan: Optional[dict[str, Any]] = None,
        replan_reason: str = "",
    ) -> dict[str, Any]:
        constraint_block = "\n".join(f"- {c}" for c in instruction_constraints) or "- (none)"
        system = (
            "You are the Dispatcher in a multi-agent ops team. You do NOT execute actions. "
            "You decompose the mission into per-truck assignments for Driver agents. "
            "Telemetry may be partial; if so, set `telemetry_first: true` and the drivers will "
            "run check_network before committing routes. Pick the cheapest viable route_hint "
            "given known transit times. Respect VIP priority on hard tasks."
            f"\n\nHard constraints:\n{constraint_block}"
            f"\n\nPlan schema (JSON only): {DISPATCHER_SCHEMA}"
        )
        memory_block = (
            f"\n\nPast graded runs (tail):\n{memory_tail}\n" if memory_tail.strip() else ""
        )
        replan_block = ""
        if prior_plan or replan_reason:
            replan_block = (
                f"\n\nThis is a RE-PLAN. Reason: {replan_reason or 'world changed'}.\n"
                f"Prior plan: {json.dumps(prior_plan or {}, separators=(',', ':'))}\n"
                "Update assignments to react to the change; preserve in-flight progress where it makes sense."
            )
        user = (
            f"Mission brief: {mission_brief}\n"
            f"Public telemetry: {json.dumps(public_state)}{memory_block}{replan_block}\n"
            "Return ONLY the plan JSON."
        )

        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        try:
            plan = json.loads(resp.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            plan = {}

        plan.setdefault("assignments", [])
        plan.setdefault("telemetry_first", True)
        plan.setdefault("rationale", "")
        cleaned: list[dict[str, Any]] = []
        for a in plan["assignments"]:
            if not isinstance(a, dict):
                continue
            tid = str(a.get("truck_id") or "")
            if not tid:
                continue
            cleaned.append(
                {
                    "truck_id": tid,
                    "order_id": a.get("order_id"),
                    "priority": int(a.get("priority") or 1),
                    "route_hint": [str(r) for r in (a.get("route_hint") or []) if r],
                    "load_amount": int(a.get("load_amount") or 0),
                }
            )
        plan["assignments"] = cleaned
        return plan
