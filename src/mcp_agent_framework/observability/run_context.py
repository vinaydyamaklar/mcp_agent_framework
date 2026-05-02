"""
RunContext — carries a run_id and optional tracer through a single agent invocation.

Each top-level pattern call creates a RunContext. Sub-patterns call `.child()`
to get a new context that shares the same tracer but has its own run_id and
records the parent linkage so traces form a tree.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from mcp_agent_framework.observability.tracer import BaseTracer, TraceEvent, TraceEventType


@dataclass
class RunContext:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_run_id: str | None = None
    tracer: BaseTracer | None = None

    async def emit(self, event_type: TraceEventType, data: dict[str, Any]) -> None:
        """Emit a trace event if a tracer is attached. No-op otherwise."""
        if self.tracer:
            await self.tracer.on_event(TraceEvent(
                event_type=event_type,
                run_id=self.run_id,
                parent_run_id=self.parent_run_id,
                timestamp=time.monotonic(),
                data=data,
            ))

    def child(self) -> "RunContext":
        """
        Create a child context for sub-patterns.

        The child gets a fresh run_id, records this context's run_id as its
        parent_run_id, and shares the same tracer — so every event in the
        sub-pattern is linked back to the parent in the trace tree.
        """
        return RunContext(parent_run_id=self.run_id, tracer=self.tracer)
