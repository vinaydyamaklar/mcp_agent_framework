"""
Shared schema utilities used by all LLM client implementations.

Extracts a plain JSON Schema dict from whatever the caller provides:
  - Pydantic v2 model class  → model.model_json_schema()
  - Pydantic v1 model class  → model.schema()
  - Plain dict               → returned as-is (assumed to already be JSON Schema)

All three LLM clients import schema_from() from here — no duplication.
"""
from __future__ import annotations

from typing import Any


class StructuredOutputError(RuntimeError):
    """Raised when a model fails to return the requested structured output."""


def schema_from(response_schema: type | dict[str, Any]) -> dict[str, Any]:
    """
    Normalise response_schema to a plain JSON Schema dict.

    Args:
        response_schema: A Pydantic model class (v1 or v2) or a JSON Schema dict.

    Returns:
        JSON Schema dict.

    Raises:
        TypeError: if response_schema is neither a Pydantic model nor a dict.

    Usage:
        class Article(BaseModel):
            title: str
            summary: str
            tags: list[str]

        schema = schema_from(Article)
        # → {"type": "object", "properties": {"title": ...}, "required": [...]}

        schema = schema_from({"type": "object", "properties": {...}})
        # → returned as-is
    """
    if isinstance(response_schema, dict):
        return response_schema

    # Pydantic v2
    if hasattr(response_schema, "model_json_schema"):
        return response_schema.model_json_schema()

    # Pydantic v1
    if hasattr(response_schema, "schema") and callable(response_schema.schema):
        return response_schema.schema()

    raise TypeError(
        f"response_schema must be a Pydantic model class or a JSON Schema dict. "
        f"Got: {type(response_schema)}"
    )
