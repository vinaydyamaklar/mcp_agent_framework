"""
Observability package for mcp_agent_framework.

Public API:
    RunContext      — carries run_id + tracer through an agent invocation
    TraceEvent      — immutable record of a single lifecycle event
    TraceEventType  — enum of all observable event kinds
    BaseTracer      — abstract base; subclass to ship events anywhere
    LoggingTracer   — default tracer that writes to Python logging at DEBUG
"""
from mcp_agent_framework.observability.tracer import (
    BaseTracer,
    LoggingTracer,
    TraceEvent,
    TraceEventType,
)
from mcp_agent_framework.observability.run_context import RunContext

__all__ = [
    "RunContext",
    "TraceEvent",
    "TraceEventType",
    "BaseTracer",
    "LoggingTracer",
]
