"""Tiny in-process message bus for inter-Driver coordination.

The bus is owned by the orchestrator (`inference_multi.py`); it is *not* part
of the OpenEnv HTTP API. Drivers may emit a side-channel `broadcast` field on
their JSON output; the orchestrator strips it before posting the action to
`/step` and replays the most recent messages into peer Drivers' prompts on the
next turn. Each broadcast also surfaces in stdout as `[MSG] from=… body=…` so
validators and reviewers can see the coordination unfold.
"""

from __future__ import annotations

from collections import deque
from typing import Optional


_MAX_BODY_LEN = 240


class MessageBus:
    """A bounded FIFO of `(sender, body)` tuples shared across Drivers."""

    def __init__(self, max_messages: int = 16) -> None:
        self._messages: deque[tuple[str, str]] = deque(maxlen=max(2, int(max_messages)))

    def post(self, sender: str, body: str) -> Optional[tuple[str, str]]:
        """Add a message; ignored if `body` is empty after stripping."""
        if not isinstance(body, str):
            return None
        clean = body.strip()
        if not clean:
            return None
        if len(clean) > _MAX_BODY_LEN:
            clean = clean[: _MAX_BODY_LEN - 1] + "…"
        msg = (str(sender), clean)
        self._messages.append(msg)
        return msg

    def recent(self, exclude_sender: Optional[str] = None, limit: int = 4) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for sender, body in reversed(self._messages):
            if exclude_sender is not None and sender == exclude_sender:
                continue
            out.append((sender, body))
            if len(out) >= limit:
                break
        return list(reversed(out))

    def all(self) -> list[tuple[str, str]]:
        return list(self._messages)
