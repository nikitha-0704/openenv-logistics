import copy

# ==========================================
# 1. THE OPS-LEAD SCENARIOS
# ==========================================
# These strings are used by the LLM to understand the "flavor" of the crisis.
SCENARIOS = {
    "easy": "OPERATION FLASH FLOOD: Route North_to_East is underwater. Your objective is to deliver 20 units to East via the South hub detour before the 18-hour deadline.",
    "medium": "OPERATION BUDGET SQUEEZE: Inventory at East is depleted. Transfer 50 units from North. Management has capped your budget at $300—find the most cost-effective path.",
    "hard": "OPERATION CASCADE FAILURE: A regional port strike has closed South_to_East. You have two orders: a 40-unit VIP delivery to East and a 10-unit standard drop at South. Prioritize the VIP SLA while managing a $500 budget."
}

def get_task_setup(task_level: str, base_state: dict) -> dict:
    """Injects high-stakes logistics crises into the baseline state."""
    state = copy.deepcopy(base_state)
    
    # Assign a professional Incident ID
    state["incident_id"] = f"INC-2026-{task_level.upper()[:3]}-01"
    
    if task_level == "easy":
        # Problem: Direct route closed. Solution: North -> South -> East.
        state["routes"]["North_to_East"]["status"] = "closed"
        state["environment_alerts"].append(f"[{state['incident_id']}] CRITICAL: Flash flooding on N-E Highway. Route CLOSED.")
        state["active_orders"] = [
            {"order_id": "ORD-EASY-01", "destination": "East", "quantity": 20, "deadline": 18, "fulfilled": False}
        ]

    elif task_level == "medium":
        # Problem: Zero stock at destination and a very tight $300 budget.
        state["budget"] = 300.0 
        state["warehouses"]["East"]["inventory"] = 0
        state["active_orders"] = [
            {"order_id": "ORD-MED-01", "destination": "East", "quantity": 50, "deadline": 48, "fulfilled": False}
        ]

    elif task_level == "hard":
        # Problem: Strike + Multiple Orders + VIP Priority.
        state["budget"] = 500.0
        state["routes"]["South_to_East"]["status"] = "closed"
        state["environment_alerts"].append(f"[{state['incident_id']}] PORT STRIKE: South Hub routes restricted. Expect heavy delays.")
        state["active_orders"] = [
            {"order_id": "ORD-HARD-VIP", "destination": "East", "quantity": 40, "deadline": 20, "fulfilled": False, "is_vip": True},
            {"order_id": "ORD-HARD-LOW", "destination": "South", "quantity": 10, "deadline": 30, "fulfilled": False, "is_vip": False}
        ]
        
    return state

# ==========================================
# 2. THE DETERMINISTIC GRADERS (raw [0,1], reported in open interval (0,1))
# ==========================================
def open_unit_score(raw: float) -> float:
    """Map a raw score in [0, 1] to (0, 1) exclusive (hackathon / LiteLLM validators)."""
    raw = max(0.0, min(1.0, float(raw)))
    lo, hi = 1e-3, 1.0 - 1e-3
    return round(lo + raw * (hi - lo), 4)


class LogisticsGrader:
    @staticmethod
    def evaluate(task_level: str, final_state: dict) -> float:
        """Evaluates agent performance based on mission-specific KPIs."""
        orders = final_state.get("active_orders", [])
        time_elapsed = final_state.get("current_time_hours", 0)

        if not orders:
            return open_unit_score(0.0)

        if task_level == "easy":
            order = orders[0]
            if order["fulfilled"]:
                # Raw 1.0 for efficiency (under 14h), 0.8 for just making the deadline.
                raw = 1.0 if time_elapsed <= 14 else 0.8
                return open_unit_score(raw)
            return open_unit_score(0.0)

        if task_level == "medium":
            order = orders[0]
            if not order["fulfilled"]:
                return open_unit_score(0.0)
            budget_remaining = max(0, final_state["budget"])
            budget_score = (budget_remaining / 300.0) * 0.5
            raw = round(0.5 + budget_score, 2)
            return open_unit_score(raw)

        if task_level == "hard":
            vip_order = next((o for o in orders if o.get("is_vip")), None)
            std_order = next((o for o in orders if not o.get("is_vip")), None)

            score = 0.0
            if vip_order and vip_order["fulfilled"]:
                score += 0.7
            if std_order and std_order["fulfilled"]:
                score += 0.3

            if final_state["budget"] < 0:
                score -= 0.4

            raw = max(0.0, min(1.0, round(score, 2)))
            return open_unit_score(raw)

        return open_unit_score(0.0)