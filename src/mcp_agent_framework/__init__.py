"""
mcp-agent-framework
===================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL REGISTRY  — register once, call anywhere, swap any time
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from mcp_agent_framework import ModelRegistry, AnthropicClient, OpenAIClient, GeminiClient

    registry = ModelRegistry()
    registry.register("fast",  AnthropicClient("claude-haiku-4-5-20251001"), tags=["cheap"])
    registry.register("smart", AnthropicClient("claude-sonnet-4-6"),         tags=["balanced"])
    registry.register("best",  AnthropicClient("claude-opus-4-6"),           tags=["powerful"])
    registry.register("gpt",   OpenAIClient("gpt-5.4"))
    registry.register("grok",  OpenAIClient("grok-3", base_url="https://api.x.ai/v1"))
    registry.register("flash", GeminiClient("gemini-2.5-flash"),             tags=["cheap"])

    # Single call — auto_execute=False (default): tool_calls returned to you
    response = await registry.complete("smart", messages, tools=tools)
    if response.has_tool_calls:
        messages.append(response.to_message())
        for tc in response.tool_calls:
            result = await mcp.call_tool(tc.name, tc.arguments)
            messages.append(tc.to_tool_result_message(result))

    # Full loop — auto_execute=True: registry runs the tool loop for you
    response = await registry.complete(
        "smart", messages, tools=tools,
        auto_execute=True,
        tool_executor=lambda name, args: mcp.call_tool(name, args),
    )
    print(response.text)

    # Structured output — guaranteed typed response from any provider
    result = await registry.complete_structured("smart", messages, MyPydanticModel)
    obj = MyPydanticModel(**result.data)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLIENTS  — provider adapters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    AnthropicClient  — Claude (claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5)
    OpenAIClient     — GPT-5.4 + Grok + Ollama + any OpenAI-compatible endpoint
    GeminiClient     — Gemini 2.5 Pro / Flash

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERNS  — agent coordination strategies
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SingleAgentLoop            — ReAct: one LLM + one MCP server
    OrchestratorWorkerPattern  — one LLM across multiple MCP workers
    HierarchicalAgentPattern   — parent delegates to child agent loops
    HumanInLoopPattern         — approval gate before tool execution
    EvaluatorOptimizerPattern  — write → evaluate → rewrite loop
    PlannerExecutorPattern     — plan then execute step by step
    ParallelPattern            — fan-out tasks, gather results

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION  — quality scoring for EvaluatorOptimizer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    AbstractEvaluator   — implement your own evaluator
    LLMEvaluator        — score content with an LLM (0–10 → 0–1 normalised)
    RubricEvaluator     — weighted criteria with optional rule-based checkers
    EvaluationResult    — score, passed, feedback, details

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEMORY  — three memory types, all in-memory by default
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SemanticMemory   — cosine similarity search (plug in your embedding fn)
    EpisodicMemory   — timestamped event log with recency + keyword scoring
    ProceduralMemory — task→procedure lookup via Jaccard similarity
    MemoryEntry      — id, content, metadata, created_at, embedding

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESILIENCE  — production reliability
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    RetryPolicy      — exponential backoff with jitter
    CircuitBreaker   — CLOSED/OPEN/HALF_OPEN state machine

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OBSERVABILITY  — trace every LLM call and tool execution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    RunContext       — run_id, parent_run_id, tracer; pass to any pattern
    LoggingTracer    — default tracer — logs to Python logging at DEBUG
    BaseTracer       — subclass to build custom tracers (OTEL, Datadog, etc.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SKILLS  — named, reusable agentic capabilities
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Skill            — name, description, input_schema, async handler
    SkillRegistry    — register / list / invoke skills by name
    SkillAwareAgent  — agent that auto-wires the registry as MCP tools

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SERVER SIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    MCPServerBase       — subclass to register tools/resources/prompts/memory
                          pass context=<any> for per-request state (self.ctx)
    StdioTransport      — client spawns server as subprocess
    HttpTransport       — standalone HTTP service (production)
    build_composed_server — merge multiple MCP servers into one endpoint

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTALL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    pip install -r requirements.txt
    pip install -e .
"""

# Types
from mcp_agent_framework.types import (
    AgentConfig, LLMResponse, MCPTool, Message, StopReason, StreamEvent, ToolCall,
)

# Clients
from mcp_agent_framework.clients import (
    BaseLLMClient, AnthropicClient, OpenAIClient, GeminiClient,
    StructuredOutputError, schema_from,
)

# Registry
from mcp_agent_framework.registry import (
    ModelRegistry, ModelResponse, StructuredModelResponse, ModelEntry,
    UsageInfo, ModelNotFoundError, ModelRegistryError, MaxIterationsError,
)

# Patterns — core
from mcp_agent_framework.patterns import (
    SingleAgentLoop,
    OrchestratorWorkerPattern, WorkerConfig,
    HierarchicalAgentPattern, ChildAgentConfig,
    HumanInLoopPattern,
    EvaluatorOptimizerPattern,
    PlannerExecutorPattern, ExecutionPlan, ExecutionStep,
    ParallelPattern, ParallelTask, ParallelResult,
)

# Patterns — evaluation
from mcp_agent_framework.patterns.evaluation import (
    AbstractEvaluator, EvaluationResult,
    LLMEvaluator,
    RubricEvaluator, RubricCriterion,
)

# Memory
from mcp_agent_framework.memory import (
    AbstractMemoryStore, MemoryEntry,
    SemanticMemory, EpisodicMemory, ProceduralMemory,
)

# Resilience
from mcp_agent_framework.resilience import (
    RetryPolicy, RetryExhaustedError,
    CircuitBreaker, CircuitState, CircuitOpenError,
)

# Observability
from mcp_agent_framework.observability import (
    RunContext, TraceEvent, TraceEventType,
    BaseTracer, LoggingTracer,
)

# RAG
from mcp_agent_framework.rag import (
    RecursiveTextChunker,
    RAGStore,
    AgenticRAGStore,
    RetrievalResult,
)

# Skills
from mcp_agent_framework.skills import (
    Skill, SkillRegistry, SkillAwareAgent,
)

# Server
from mcp_agent_framework.server import (
    MCPServerBase, ExampleServer,
    build_composed_server,
    StdioTransport, HttpTransport,
)

__all__ = [
    # types
    "AgentConfig", "LLMResponse", "MCPTool", "Message", "StopReason", "StreamEvent", "ToolCall",
    # clients
    "BaseLLMClient", "AnthropicClient", "OpenAIClient", "GeminiClient",
    "StructuredOutputError", "schema_from",
    # registry
    "ModelRegistry", "ModelResponse", "StructuredModelResponse", "ModelEntry",
    "UsageInfo", "ModelNotFoundError", "ModelRegistryError", "MaxIterationsError",
    # patterns — core
    "SingleAgentLoop",
    "OrchestratorWorkerPattern", "WorkerConfig",
    "HierarchicalAgentPattern", "ChildAgentConfig",
    "HumanInLoopPattern",
    "EvaluatorOptimizerPattern",
    "PlannerExecutorPattern", "ExecutionPlan", "ExecutionStep",
    "ParallelPattern", "ParallelTask", "ParallelResult",
    # patterns — evaluation
    "AbstractEvaluator", "EvaluationResult",
    "LLMEvaluator",
    "RubricEvaluator", "RubricCriterion",
    # memory
    "AbstractMemoryStore", "MemoryEntry",
    "SemanticMemory", "EpisodicMemory", "ProceduralMemory",
    # rag
    "RecursiveTextChunker",
    "RAGStore",
    "AgenticRAGStore",
    "RetrievalResult",
    # resilience
    "RetryPolicy", "RetryExhaustedError",
    "CircuitBreaker", "CircuitState", "CircuitOpenError",
    # observability
    "RunContext", "TraceEvent", "TraceEventType",
    "BaseTracer", "LoggingTracer",
    # skills
    "Skill", "SkillRegistry", "SkillAwareAgent",
    # server
    "MCPServerBase", "ExampleServer",
    "build_composed_server",
    "StdioTransport", "HttpTransport",
]
