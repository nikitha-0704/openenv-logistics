import copy
from typing import Tuple
from models import LogisticsAction, LogisticsObservation, RewardInfo
from tasks import get_task_setup

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

class LogisticsEnv:
    def __init__(self):
        self._current_state = {}
        self.check_count = 0
        self.task_level = "easy"
        self.reset()

    def reset(self, task_level: str = "easy") -> dict:
        """Reset the environment state and inject task-specific crises."""
        self.task_level = task_level
        self.check_count = 0
        base_state = copy.deepcopy(DEFAULT_STATE)
        self._current_state = get_task_setup(task_level, base_state)
        return self._current_state

    def state(self) -> dict:
        return self._current_state

    def step(self, action: LogisticsAction) -> Tuple[LogisticsObservation, float, bool, dict]:
        state = self._current_state
        reward = 0.0
        done = False
        obs_message = ""
        obs_success = False
        obs_data = None

        # --- ACTION 1: CHECK NETWORK (With Anti-Farming & Structured Data) ---
        if action.action_type == "check_network":
            self.check_count += 1
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
            hours = action.hours or 1
            state["current_time_hours"] += hours
            obs_success = True
            
            # Loop-aware reward
            trucks_moving = any(t["status"] == "moving" for t in state["trucks"].values())
            reward += 0.02 if trucks_moving else -0.05 # Discourage stalling

            # News Ticker flavor
            ticker = ["Fuel prices stable.", "Driver rest periods enforced.", "Satellite links clear.", "Traffic patterns normal."]
            news = ticker[state["current_time_hours"] % len(ticker)]
            obs_message = f"[SYSTEM_OK] Time +{hours}h. {news} "

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

        observation = LogisticsObservation(success=obs_success, message=obs_message.strip(), data=obs_data)
        info = RewardInfo(
            task_completed=done and any(o["fulfilled"] for o in state["active_orders"]),
            budget_remaining=state["budget"],
            penalty_incurred=abs(reward) if reward < 0 else 0.0
        ).model_dump()

        return observation, reward, done, info