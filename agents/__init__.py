"""Hierarchical multi-agent harness for the Global Logistics Resolver.

`Dispatcher` decomposes the mission into per-truck assignments under the same
partial-observability rules as the single-agent baseline. `Driver` agents
execute their assigned plan via the same OpenEnv HTTP API.

The single-agent baseline (`inference.py`) is intentionally preserved for
apples-to-apples comparison; the multi-agent harness lives in
`inference_multi.py`.
"""

from agents.bus import MessageBus
from agents.dispatcher import Dispatcher
from agents.driver import Driver
from agents.llm import MODEL_NAME, OPENENV_BASE_URL, client

__all__ = ["Dispatcher", "Driver", "MessageBus", "client", "MODEL_NAME", "OPENENV_BASE_URL"]
