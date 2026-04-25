from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any

# OpenEnv-core base classes (Pydantic models with validation + extra="forbid"
# on Action, extra="allow" on State). Subclassing them is what `openenv validate`
# expects of an OpenEnv-compliant environment.
from openenv.core import Action as _CoreAction
from openenv.core import Observation as _CoreObservation
from openenv.core import State as _CoreState

# 1. THE ACTION SPACE (What the AI can do)

class LogisticsAction(_CoreAction):
    """The strict set of tools the AI can use to solve the logistics crisis.

    Inherits OpenEnv's `Action`: Pydantic model with `extra="forbid"` (no
    silent typos in tool names) and an optional `metadata` dict.
    """

    action_type: Literal[
        "check_network",      # Look at the current state/alerts
        "load_truck",         # Put inventory into a truck
        "route_truck",        # Send a truck down a specific route
        "wait"                # Advance time (crucial for logistics!)
    ] = Field(..., description="The specific action to take.")

    # Parameters for 'load_truck'
    truck_id: Optional[str] = Field(None, description="ID of the truck (e.g., 'T101').")
    warehouse: Optional[str] = Field(None, description="Name of the warehouse (e.g., 'North').")
    amount: Optional[int] = Field(None, description="Amount of inventory to load.")

    # Parameters for 'route_truck'
    route_id: Optional[str] = Field(None, description="The route to take (e.g., 'North_to_East').")

    # Parameters for 'wait'
    hours: Optional[int] = Field(None, description="Number of hours to advance the simulation.")


# 2. THE OBSERVATION SPACE (What the AI sees)

class LogisticsObservation(_CoreObservation):
    """What the environment returns immediately after an action.

    Inherits OpenEnv's `Observation` (`done`, `reward`, `metadata` fields)
    and adds the logistics-specific success/message/data payload.
    """

    success: bool = Field(..., description="Did the action execute successfully?")
    message: str = Field(..., description="Text feedback (e.g., 'Truck T101 routed successfully' or 'Error: Route Closed').")
    data: Optional[Dict[str, Any]] = Field(None, description="JSON payload containing requested network info.")

# 3. REWARD (OpenEnv: typed reward signal from step)

class LogisticsReward(BaseModel):
    """Scalar step reward returned alongside observation and done flag."""

    value: float = Field(..., description="Step reward; may be shaped for partial progress and penalties.")

# 4. INFO (Auxiliary step metadata)

class RewardInfo(BaseModel):
    """Tracks partial progress and budget penalties."""

    task_completed: bool = Field(default=False, description="Is the primary objective met?")
    budget_remaining: float = Field(..., description="How much money is left.")
    penalty_incurred: float = Field(default=0.0, description="Penalties for invalid actions or delays.")


# 5. STATE (OpenEnv: introspectable environment state)

class LogisticsState(_CoreState):
    """Snapshot of the env's full simulation state.

    `_CoreState` allows extra fields, so we can carry the live `dict` shape
    (warehouses / routes / trucks / orders / alerts / mission_brief / …)
    without re-declaring every key while still gaining episode_id + step_count.
    """