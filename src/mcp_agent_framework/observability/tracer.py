"""
Tracer abstractions for observability.

TraceEventType enumerates all observable lifecycle events in the agent framework.
BaseTracer is the abstract sink — implement on_event to ship events anywhere
(logging, OTLP, LangSmith, custom DB, etc.).
LoggingTracer is the out-of-the-box default that writes to Python logging at DEBUG.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TraceEventType(str, Enum):
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_ERROR = "tool_error"
    PATTERN_START = "pattern_start"
    PATTERN_END = "pattern_end"


@dataclass
class TraceEvent:
    event_type: TraceEventType
    run_id: str
    timestamp: float        # time.monotonic()
    data: dict[str, Any]
    parent_run_id: str | None = None


class BaseTracer(ABC):
    """Abstract sink for trace events. Override on_event to ship events anywhere."""

    @abstractmethod
    async def on_event(self, event: TraceEvent) -> None:
        """Receive a single trace event. Implementations should be non-blocking."""
        ...

    # ------------------------------------------------------------------
    # Convenience methods — each builds a TraceEvent then calls on_event.
    # ------------------------------------------------------------------

    async def on_llm_start(
        self,
        run_id: str,
        model: str,
        messages: list[Any],
        **kw: Any,
    ) -> None:
        await self.on_event(TraceEvent(
            event_type=TraceEventType.LLM_START,
            run_id=run_id,
            timestamp=time.monotonic(),
            data={"model": model, "messages": messages, **kw},
        ))

    async def on_llm_end(
        self,
        run_id: str,
        model: str,
        response_content: str,
        elapsed_ms: float,
        **kw: Any,
    ) -> None:
        await self.on_event(TraceEvent(
            event_type=TraceEventType.LLM_END,
            run_id=run_id,
            timestamp=time.monotonic(),
            data={
                "model": model,
                "response_content": response_content,
                "elapsed_ms": elapsed_ms,
                **kw,
            },
        ))

    async def on_tool_start(
        self,
        run_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        **kw: Any,
    ) -> None:
        await self.on_event(TraceEvent(
            event_type=TraceEventType.TOOL_START,
            run_id=run_id,
            timestamp=time.monotonic(),
            data={"tool_name": tool_name, "arguments": arguments, **kw},
        ))

    async def on_tool_end(
        self,
        run_id: str,
        tool_name: str,
        result: Any,
        elapsed_ms: float,
        **kw: Any,
    ) -> None:
        await self.on_event(TraceEvent(
            event_type=TraceEventType.TOOL_END,
            run_id=run_id,
            timestamp=time.monotonic(),
            data={
                "tool_name": tool_name,
                "result": result,
                "elapsed_ms": elapsed_ms,
                **kw,
            },
        ))

    async def on_tool_error(
        self,
        run_id: str,
        tool_name: str,
        error: Exception | str,
        **kw: Any,
    ) -> None:
        await self.on_event(TraceEvent(
            event_type=TraceEventType.TOOL_ERROR,
            run_id=run_id,
            timestamp=time.monotonic(),
            data={"tool_name": tool_name, "error": str(error), **kw},
        ))

    async def on_pattern_start(
        self,
        run_id: str,
        pattern_name: str,
        user_message: str,
        **kw: Any,
    ) -> None:
        await self.on_event(TraceEvent(
            event_type=TraceEventType.PATTERN_START,
            run_id=run_id,
            timestamp=time.monotonic(),
            data={"pattern_name": pattern_name, "user_message": user_message, **kw},
        ))

    async def on_pattern_end(
        self,
        run_id: str,
        pattern_name: str,
        result: Any,
        elapsed_ms: float,
        **kw: Any,
    ) -> None:
        await self.on_event(TraceEvent(
            event_type=TraceEventType.PATTERN_END,
            run_id=run_id,
            timestamp=time.monotonic(),
            data={
                "pattern_name": pattern_name,
                "result": result,
                "elapsed_ms": elapsed_ms,
                **kw,
            },
        ))


class LoggingTracer(BaseTracer):
    """
    Default tracer — emits every event to Python logging at DEBUG level.

    Format: [trace] {event_type} | run={run_id[:8]} | data={data}
    """

    async def on_event(self, event: TraceEvent) -> None:
        logger.debug(
            "[trace] %s | run=%s | data=%s",
            event.event_type,
            event.run_id[:8],
            event.data,
        )
