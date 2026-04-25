import copy
import os
from typing import Any, Optional, Tuple

from openenv.core import Environment as _CoreEnvironment

from models import LogisticsAction, LogisticsObservation, LogisticsState, RewardInfo
from tasks import LogisticsGrader, get_task_setup

# Adversarial mode (ADVERSARIAL=1): once, on the first `wait` action AFTER the env
# has processed at least `ADVERSARIAL_TRIGGER_STEP` actions, fire a surprise
# infrastructure alert and re-lock telemetry. In-flight trucks finish their
# current dispatch (the route-closed check happens at next dispatch), so
# already-routed plans are not retroactively cancelled — the goal is to test
# whether the agent re-syncs and re-plans, not to gotcha-fail it.
#
# Per task: `easy` is a strict 2-route detour problem, so closing either East-
# bound route would be unrecoverable. We emit an alert + re-lock telemetry only.
# `medium` and `hard` close the primary North->East corridor, forcing a longer
# 2-hop detour for any *future* dispatch.
ADVERSARIAL_CLOSURE = {
    "easy": None,                 # alert + telemetry re-lock only (recoverable)
    "medium": "North_to_East",    # forces costlier two-hop path next time
    "hard": "North_to_East",      # cuts the VIP express corridor next time
}
ADVERSARIAL_TRIGGER_STEP = 3


def _adversarial_enabled() -> bool:
    return os.getenv("ADVERSARIAL", "").strip().lower() in {"1", "true", "yes", "on"}

DEFAULT_STATE = {
    "budget": 5000.0,
    "current_time_hours": 0,
    "warehouses": {
        "North": {"inventory": 100},
        "South": {"inventory": 0}, 
        "East":  {"inventory": 0}
    },
    "routes": {
        "North_to_East": {"transit_time": 10, "cost": 200, "status": "open", "from": "North", "to": "East"},
        "North_to_South": {"transit_time": 5, "cost": 100, "status": "open", "from": "North", "to": "South"},
        "South_to_East": {"transit_time": 8, "cost": 150, "status": "open", "from": "South", "to": "East"},
        "South_to_North": {"transit_time": 5, "cost": 100, "status": "open", "from": "South", "to": "North"} 
    },
    "trucks": {
        "T101": {"location": "North", "destination": None, "cargo": 0, "status": "idle", "time_remaining": 0},
        "T102": {"location": "South", "destination": None, "cargo": 0, "status": "idle", "time_remaining": 0}
    },
    "active_orders": [], "environment_alerts": []
}

class LogisticsEnv(_CoreEnvironment[LogisticsAction, LogisticsObservation, LogisticsState]):
    """OpenEnv-compliant logistics environment.

    Subclasses `openenv.core.Environment[LogisticsAction, LogisticsObservation,
    LogisticsState]`. The HTTP shim (``server/app.py``) and the historical CLI
    callers expect:
      - ``reset(task_level=...) -> dict``      (legacy "full state" return)
      - ``step(action) -> (obs, reward, done, info)``  (legacy 4-tuple)
      - ``state() -> dict``                    (legacy method-style accessor)
    All three are preserved here. The base ABC requires methods named
    ``reset``/``step``/``state`` to exist on the subclass; we satisfy that and
    additionally bind a typed ``rubric`` (``LogisticsGrader.rubric_for``) so
    training infrastructure can introspect the grader composition via
    ``env.rubric.named_rubrics()``.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        super().__init__(rubric=LogisticsGrader.rubric_for("easy"))
        self._current_state: dict = {}
        self.check_count = 0
        self._telemetry_unlocked = False
        self._step_count = 0
        self._adversarial_fired = False
        self.task_level = "easy"
        self.reset()

    def reset(
        self,
        task_level: str = "easy",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Reset the environment state and inject task-specific crises.

        OpenEnv-style ``seed``/``episode_id`` are accepted for compatibility
        with ``Environment.reset_async`` and the platform validator; they're
        recorded on the state dict but the env itself is deterministic.
        """
        self.task_level = task_level
        self.check_count = 0
        self._telemetry_unlocked = False
        self._step_count = 0
        self._adversarial_fired = False
        base_state = copy.deepcopy(DEFAULT_STATE)
        self._current_state = get_task_setup(task_level, base_state)
        if episode_id is not None:
            self._current_state["episode_id"] = episode_id
        if seed is not None:
            self._current_state["seed"] = int(seed)
        # Re-bind the rubric to the active task so `env.rubric` is always the
        # right grader (handy for trainers calling `env.rubric(action, state)`).
        self.rubric = LogisticsGrader.rubric_for(task_level)
        return self._current_state

    def _maybe_fire_adversarial(self, state: dict) -> str:
        """Fire one adversarial event; return any extra observation suffix."""
        if self._adversarial_fired or not _adversarial_enabled():
            return ""
        if self._step_count <= ADVERSARIAL_TRIGGER_STEP:
            return ""
        self._adversarial_fired = True
        closure = ADVERSARIAL_CLOSURE.get(self.task_level)
        alert_id = f"INC-2026-{self.task_level.upper()[:3]}-ADV"
        suffix_parts = []
        if closure and state["routes"].get(closure, {}).get("status") == "open":
            state["routes"][closure]["status"] = "closed"
            state["environment_alerts"].append(
                f"[{alert_id}] BREAKING: Route {closure} CLOSED — secondary incident."
            )
            suffix_parts.append(
                f"[BREAKING] {alert_id}: Route {closure} just closed. Re-sync telemetry."
            )
        else:
            state["environment_alerts"].append(
                f"[{alert_id}] BREAKING: Regional advisory — telemetry de-synced."
            )
            suffix_parts.append(
                f"[BREAKING] {alert_id}: Telemetry de-synced. Re-sync before next dispatch."
            )
        # Re-lock public telemetry: agent must call `check_network` again to see truth.
        self._telemetry_unlocked = False
        return " " + " ".join(suffix_parts)

    def state(self) -> dict:
        """Full internal state (used by grader and physics).

        Kept method-shaped (not a property) for back-compat with `server/app.py`
        and `test_local.py`. The OpenEnv ABC only requires the *attribute*
        ``state`` to exist on the subclass — a callable satisfies that.
        """
        return self._current_state

    def state_model(self) -> LogisticsState:
        """Typed ``LogisticsState`` snapshot — convenient for trainers and
        the OpenEnv multi-mode validator.
        """
        return LogisticsState(
            episode_id=self._current_state.get("episode_id"),
            step_count=self._step_count,
            **{k: v for k, v in self._current_state.items() if k != "episode_id"},
        )

    def public_state(self) -> dict:
        """World-modeling: agent-facing partial observation until check_network unlocks full telemetry."""
        s = copy.deepcopy(self._current_state)
        if self._telemetry_unlocked:
            s["telemetry_visibility"] = "full"
            return s
        s["telemetry_visibility"] = "partial"
        masked_routes = {}
        for rid, r in s.get("routes", {}).items():
            masked_routes[rid] = {
                "from": r["from"],
                "to": r["to"],
                "transit_time": r.get("transit_time"),
                "status": "unknown",
                "cost": None,
            }
        s["routes"] = masked_routes
        masked_orders = []
        for o in s.get("active_orders", []):
            od = {k: v for k, v in o.items() if k != "deadline"}
            od["deadline"] = None
            od["deadline_note"] = "Run check_network for SLA precision."
            masked_orders.append(od)
        s["active_orders"] = masked_orders
        return s

    def step(
        self,
        action: LogisticsAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[LogisticsObservation, float, bool, dict]:
        state = self._current_state
        self._step_count += 1
        reward = 0.0
        done = False
        obs_message = ""
        obs_success = False
        obs_data = None

        # --- ACTION 1: CHECK NETWORK (With Anti-Farming & Structured Data) ---
        if action.action_type == "check_network":
            self.check_count += 1
            self._telemetry_unlocked = True
            obs_success = True
            obs_message = "[BUSINESS_EVENT] Full network telemetry synchronized."
            
            # Tiny reward for first few checks; zero thereafter to prevent loops
            reward += 0.05 if self.check_count < 3 else 0.0
            
            obs_data = {
                "incident_id": state.get("incident_id"),
                "route_status": {r_id: r["status"] for r_id, r in state["routes"].items()},
                "sla_countdown": {
                    o["order_id"]: f"{o['deadline'] - state['current_time_hours']}h remaining" 
                    for o in state["active_orders"]
                }
            }

        # --- ACTION 2: LOAD TRUCK ---
        elif action.action_type == "load_truck":
            truck = state["trucks"].get(action.truck_id)
            warehouse = state["warehouses"].get(action.warehouse)
            
            if not truck or not warehouse or truck["location"] != action.warehouse or truck["status"] != "idle":
                obs_message = f"[VALIDATION_ERROR] Asset {action.truck_id} unavailable at {action.warehouse}."
                reward -= 0.1
            elif warehouse["inventory"] < (action.amount or 0):
                obs_message = f"[VALIDATION_ERROR] Insufficient inventory at {action.warehouse}."
                reward -= 0.1
            else:
                warehouse["inventory"] -= action.amount
                truck["cargo"] += action.amount
                obs_success = True
                obs_message = f"Loaded {action.amount} units onto {action.truck_id}."
                reward += 0.2

        # --- ACTION 3: ROUTE TRUCK ---
        elif action.action_type == "route_truck":
            truck = state["trucks"].get(action.truck_id)
            route = state["routes"].get(action.route_id)

            if not truck or not route or truck["status"] != "idle" or truck["location"] != route["from"]:
                obs_message = f"[VALIDATION_ERROR] Invalid dispatch for {action.truck_id}."
                reward -= 0.1
            elif route["status"] != "open":
                obs_message = f"[CRITICAL_FAILURE] Route {action.route_id} is CLOSED. Dispatch aborted."
                reward -= 0.3 # Harsh penalty for ignoring alerts
            elif state["budget"] < route["cost"]:
                obs_message = "[BUDGET_EXCEEDED] Insufficient funds for route cost."
                reward -= 0.2
            else:
                state["budget"] -= route["cost"]
                truck.update({
                    "status": "moving",
                    "destination": route["to"],
                    "time_remaining": route["transit_time"],
                    "location": "in_transit"
                })
                obs_success = True
                obs_message = f"Dispatched {action.truck_id} via {action.route_id}. Budget impact: -${route['cost']}."
                reward += 0.3

        # --- ACTION 4: WAIT (With News Ticker & Stalling Penalties) ---
        elif action.action_type == "wait":
            adversarial_suffix = self._maybe_fire_adversarial(state)

            hours = action.hours or 1
            state["current_time_hours"] += hours
            obs_success = True

            # Loop-aware reward
            trucks_moving = any(t["status"] == "moving" for t in state["trucks"].values())
            reward += 0.02 if trucks_moving else -0.05 # Discourage stalling

            # News Ticker flavor
            ticker = ["Fuel prices stable.", "Driver rest periods enforced.", "Satellite links clear.", "Traffic patterns normal."]
            news = ticker[state["current_time_hours"] % len(ticker)]
            obs_message = f"[SYSTEM_OK] Time +{hours}h. {news}{adversarial_suffix} "

            # Process Arrivals
            for t_id, t_data in state["trucks"].items():
                if t_data["status"] == "moving":
                    t_data["time_remaining"] -= hours
                    if t_data["time_remaining"] <= 0:
                        dest, cargo = t_data["destination"], t_data["cargo"]
                        state["warehouses"][dest]["inventory"] += cargo
                        t_data.update({"status": "idle", "location": dest, "destination": None, "cargo": 0, "time_remaining": 0})
                        obs_message += f"Truck {t_id} arrived at {dest}. "

            # Order Resolution
            for order in state["active_orders"]:
                if not order["fulfilled"]:
                    dest_warehouse = state["warehouses"][order["destination"]]
                    if dest_warehouse["inventory"] >= order["quantity"] and state["current_time_hours"] <= order["deadline"]:
                        order["fulfilled"] = True
                        dest_warehouse["inventory"] -= order["quantity"]
                        obs_message += f"Order {order['order_id']} FULFILLED! "
                        reward += 1.0
                    elif state["current_time_hours"] > order["deadline"]:
                        obs_message += f"Order {order['order_id']} FAILED (Deadline missed). "
                        reward -= 1.0

        # --- EPISODE TERMINATION LOGIC ---
        if self.task_level == "hard":
            # Ends only when all orders are resolved (Success or Fail)
            done = all(o["fulfilled"] or state["current_time_hours"] > o["deadline"] for o in state["active_orders"])
        else:
            # Easy/Medium ends on first fulfillment or any failure
            done = any(o["fulfilled"] or state["current_time_hours"] > o["deadline"] for o in state["active_orders"])

        state["network_sync_count"] = self.check_count

        observation = LogisticsObservation(
            success=obs_success,
            message=obs_message.strip(),
            data=obs_data,
            done=done,
            reward=reward,
        )
        info = RewardInfo(
            task_completed=done and any(o["fulfilled"] for o in state["active_orders"]),
            budget_remaining=state["budget"],
            penalty_incurred=abs(reward) if reward < 0 else 0.0
        ).model_dump()

        return observation, reward, done, info