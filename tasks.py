import copy
from typing import Any, Optional

from openenv.core.rubrics import Rubric, WeightedSum

# ==========================================
# 1. THE OPS-LEAD SCENARIOS
# ==========================================
# These strings are used by the LLM to understand the "flavor" of the crisis.
SCENARIOS = {
    "easy": "OPERATION FLASH FLOOD: Route North_to_East is underwater. Your objective is to deliver 20 units to East via the South hub detour before the 18-hour deadline.",
    "medium": "OPERATION BUDGET SQUEEZE: Inventory at East is depleted. Transfer 50 units from North. Management has capped your budget at $300—find the most cost-effective path.",
    "hard": "OPERATION CASCADE FAILURE: A regional port strike has closed South_to_East. You have two orders: a 40-unit VIP delivery to East and a 10-unit standard drop at South. Prioritize the VIP SLA while managing a $500 budget."
}

# Long-horizon planning + instruction-following (natural language + structured constraints).
MISSION_BRIEF = {
    "easy": (
        "You are the on-call ops lead. Flash flooding has compromised the primary corridor. "
        "Your charter: restore service to East without guessing infrastructure state. "
        "Telemetry starts partial—sync before you commit trucks to long hauls. "
        "Deliver exactly 20 units to East before the SLA expires."
    ),
    "medium": (
        "East is dry on inventory and leadership capped spend at $300. "
        "Move 50 units from North to East. Partial telemetry hides precise route costs and SLA "
        "countdowns until you run check_network; then choose the cheapest viable path."
    ),
    "hard": (
        "Port labor action has restricted the South hub. You hold two contracts: a VIP block to East "
        "and a standard replenishment to South, under a $500 budget. Coordinate both trucks; VIP "
        "must not slip while you still honor the secondary drop if funds and time allow."
    ),
}

INSTRUCTION_CONSTRAINTS = {
    "easy": [
        "Run check_network at least once before the episode ends if you intend to fulfill the order (ops discipline).",
        "Fulfill ORD-EASY-01 (20 units to East) before its deadline.",
        "Do not dispatch on routes whose status you have not verified after telemetry is partial.",
    ],
    "medium": [
        "Never exceed the $300 budget.",
        "Run check_network before locking in a high-cost route when SLA precision matters.",
        "Fulfill ORD-MED-01 (50 units to East) before its deadline.",
    ],
    "hard": [
        "Prioritize ORD-HARD-VIP (40 units to East) over the standard order when tradeoffs appear.",
        "Fulfill both orders when possible without bankrupting the mission (budget must not finish negative).",
        "Use check_network when graph status is uncertain after incident alerts.",
    ],
}


def get_task_setup(task_level: str, base_state: dict) -> dict:
    """Injects high-stakes logistics crises into the baseline state."""
    state = copy.deepcopy(base_state)
    
    # Assign a professional Incident ID
    state["incident_id"] = f"INC-2026-{task_level.upper()[:3]}-01"
    state["network_sync_count"] = 0
    state["mission_brief"] = MISSION_BRIEF.get(task_level, SCENARIOS.get(task_level, ""))
    state["instruction_constraints"] = list(INSTRUCTION_CONSTRAINTS.get(task_level, []))

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
# 2. COMPOSABLE RUBRICS (openenv.core.rubrics.Rubric)
# ==========================================
# Each atomic rubric scores one well-named signal in [0, 1] from the env's
# final state. Per-task graders (Easy/Medium/Hard) compose these atoms via
# WeightedSum + a multiplicative discipline gate. Atoms expose `last_score`
# so trainers / the eval notebook can introspect contributions per rollout:
#
#     rubric = LogisticsGrader.rubric_for("hard")
#     rubric(action=None, observation=final_state)
#     for name, child in rubric.named_rubrics():
#         print(name, round(child.last_score or 0.0, 3))
#
# `LogisticsGrader.evaluate(task_level, final_state)` keeps the historical
# open-interval (0, 1) public contract for the hackathon / LiteLLM validators.

def open_unit_score(raw: float) -> float:
    """Map a raw score in [0, 1] to (0, 1) exclusive (hackathon / LiteLLM validators)."""
    raw = max(0.0, min(1.0, float(raw)))
    lo, hi = 1e-3, 1.0 - 1e-3
    return round(lo + raw * (hi - lo), 4)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _orders(state: dict) -> list[dict]:
    return list(state.get("active_orders", []) or [])


def _find_order(state: dict, order_id: Optional[str] = None, vip: Optional[bool] = None) -> Optional[dict]:
    for o in _orders(state):
        if order_id is not None and o.get("order_id") == order_id:
            return o
        if vip is True and o.get("is_vip"):
            return o
        if vip is False and not o.get("is_vip"):
            return o
    return None


class _StateRubric(Rubric):
    """Convenience base for rubrics that look only at the final state.

    The OpenEnv Rubric API is `forward(action, observation) -> float`. For our
    terminal grader the "observation" is the env's final-state dict and the
    "action" is unused, so we expose a `score(state)` hook that subclasses
    implement.
    """

    def forward(self, action: Any, observation: Any) -> float:  # type: ignore[override]
        state = observation if isinstance(observation, dict) else {}
        return _clip01(self.score(state))

    def score(self, state: dict) -> float:  # pragma: no cover - abstract-ish
        raise NotImplementedError


class DeliveryCompletion(_StateRubric):
    """1.0 if the named order is fulfilled, else 0.0."""

    def __init__(self, order_id: Optional[str] = None, vip: Optional[bool] = None):
        super().__init__()
        self.order_id = order_id
        self.vip = vip

    def score(self, state: dict) -> float:
        order = _find_order(state, order_id=self.order_id, vip=self.vip)
        if not order:
            return 0.0
        return 1.0 if order.get("fulfilled") else 0.0


class TimeEfficiency(_StateRubric):
    """1.0 if elapsed_hours <= efficient_hours else `partial` if <= deadline_hours, else 0.

    `deadline_hours=None` disables the lower tier (returns `partial` for any t > efficient).
    """

    def __init__(self, deadline_hours: Optional[int], efficient_hours: int, partial: float = 0.8):
        super().__init__()
        self.deadline_hours = deadline_hours
        self.efficient_hours = efficient_hours
        self.partial = partial

    def score(self, state: dict) -> float:
        t = int(state.get("current_time_hours", 0) or 0)
        if t <= self.efficient_hours:
            return 1.0
        if self.deadline_hours is None or t <= self.deadline_hours:
            return float(self.partial)
        return 0.0


class InsolvencyPenalty(_StateRubric):
    """Returns `penalty` magnitude (>=0) if budget < 0 at episode end, else 0.

    Used as a *subtractive* penalty by per-task graders that compose it.
    """

    def __init__(self, penalty: float):
        super().__init__()
        self.penalty = float(penalty)

    def score(self, state: dict) -> float:
        return self.penalty if float(state.get("budget", 0.0)) < 0.0 else 0.0


class BudgetRetention(_StateRubric):
    """`max(0, budget_remaining) / cap`, clipped to [0,1]."""

    def __init__(self, cap: float):
        super().__init__()
        self.cap = float(cap)

    def score(self, state: dict) -> float:
        remaining = max(0.0, float(state.get("budget", 0.0)))
        return _clip01(remaining / max(1e-9, self.cap))


class BudgetSolvency(_StateRubric):
    """1.0 if budget >= 0 at episode end, else 0.0 — multiplicative gate for hard tasks."""

    def score(self, state: dict) -> float:
        return 1.0 if float(state.get("budget", 0.0)) >= 0.0 else 0.0


class NetworkSyncDiscipline(_StateRubric):
    """Instruction-following: returns `multiplier` (<=1) if no sync, else 1.0."""

    def __init__(self, min_syncs: int = 1, multiplier: float = 0.85):
        super().__init__()
        self.min_syncs = int(min_syncs)
        self.multiplier = float(multiplier)

    def score(self, state: dict) -> float:
        syncs = int(state.get("network_sync_count", 0) or 0)
        return 1.0 if syncs >= self.min_syncs else self.multiplier


class _TaskRubric(_StateRubric):
    """Base for per-task graders. Subclasses build a small tree of named
    children (auto-registered by `Rubric.__setattr__`) and combine them in
    `score()`. Children expose `last_score` for trainers / the eval notebook.
    """

    pass


class EasyRubric(_TaskRubric):
    """Easy: `delivery × speed × network_sync_discipline`.

    Reproduces the legacy contract exactly: 1.0 if delivered ≤14h with at
    least one telemetry sync; 0.85 if delivered ≤14h with no sync; 0.8 / 0.68
    for delivered ≤18h; 0.0 if missed deadline / not delivered.
    """

    def __init__(self):
        super().__init__()
        self.delivery = DeliveryCompletion(order_id="ORD-EASY-01")
        self.time_efficiency = TimeEfficiency(deadline_hours=18, efficient_hours=14, partial=0.8)
        self.network_sync_discipline = NetworkSyncDiscipline(min_syncs=1, multiplier=0.85)

    def score(self, state: dict) -> float:
        delivered = self.delivery(None, state)
        if delivered <= 0.0:
            self.time_efficiency(None, state)
            self.network_sync_discipline(None, state)
            return 0.0
        speed = float(self.time_efficiency(None, state))
        discipline = float(self.network_sync_discipline(None, state))
        return _clip01(delivered * speed * discipline)


class MediumRubric(_TaskRubric):
    """Medium: `(0.5 + 0.5 × budget_retention) × discipline`, gated on delivery.

    Matches the legacy `0.5 + (budget_remaining/300) * 0.5` shape via a
    `WeightedSum` of a delivery floor and budget retention.
    """

    def __init__(self):
        super().__init__()
        self.delivery = DeliveryCompletion(order_id="ORD-MED-01")
        self.budget_retention = BudgetRetention(cap=300.0)
        self.network_sync_discipline = NetworkSyncDiscipline(min_syncs=1, multiplier=0.85)
        self.kpi_blend = WeightedSum([self.delivery, self.budget_retention], [0.5, 0.5])

    def score(self, state: dict) -> float:
        if self.delivery(None, state) <= 0.0:
            self.budget_retention(None, state)
            self.network_sync_discipline(None, state)
            self.kpi_blend(None, state)
            return 0.0
        base = float(self.kpi_blend(None, state))
        discipline = float(self.network_sync_discipline(None, state))
        return _clip01(base * discipline)


class HardRubric(_TaskRubric):
    """Hard: `0.7×VIP + 0.3×standard − 0.4×insolvent`, clipped to [0,1].

    Faithful to the legacy additive-penalty contract.
    """

    def __init__(self):
        super().__init__()
        self.vip_delivery = DeliveryCompletion(vip=True)
        self.standard_delivery = DeliveryCompletion(vip=False)
        self.insolvency_penalty = InsolvencyPenalty(penalty=0.4)
        self.priority_blend = WeightedSum(
            [self.vip_delivery, self.standard_delivery], [0.7, 0.3]
        )

    def score(self, state: dict) -> float:
        base = float(self.priority_blend(None, state))
        penalty = float(self.insolvency_penalty(None, state))
        return _clip01(base - penalty)


_RUBRICS_BY_LEVEL = {
    "easy": EasyRubric,
    "medium": MediumRubric,
    "hard": HardRubric,
}


class LogisticsGrader:
    """Public grader. Implementation = composition of `openenv.core.rubrics`.

    `evaluate()` keeps the historical (0, 1) open-interval contract used by the
    HTTP `/grader` endpoint and the LiteLLM validators. `rubric_for()` returns
    the underlying composable rubric so trainers can read named sub-scores.
    """

    @staticmethod
    def rubric_for(task_level: str) -> _TaskRubric:
        factory = _RUBRICS_BY_LEVEL.get(task_level)
        if factory is None:
            class _Zero(_StateRubric):
                def score(self, state: dict) -> float:  # noqa: D401
                    return 0.0
            return _Zero()
        return factory()

    @staticmethod
    def evaluate(task_level: str, final_state: dict) -> float:
        if not _orders(final_state):
            return open_unit_score(0.0)
        rubric = LogisticsGrader.rubric_for(task_level)
        raw = float(rubric(None, final_state))
        return open_unit_score(raw)