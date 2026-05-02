"""
=============================================================================
EXAMPLE 02: Structured Output — extract typed data from free-form text
=============================================================================

WHAT IS STRUCTURED OUTPUT?
---------------------------
By default, a language model returns a blob of text. That is fine for
a chat UI, but useless if you need to store the result in a database, pass
it to another function, or display specific fields in a UI.

Structured output forces the model to return JSON that matches a schema you
define. You get a Python object with typed fields — no brittle regex or
"parse the model's prose" hacks.

  Without structured output:  "The article is by John Smith, published in
                               2024, covering AI topics..."
                               → you have to parse that yourself

  With structured output:     {"author": "John Smith", "year": 2024,
                               "topics": ["AI", "Machine Learning"]}
                               → drop it straight into your database

WHY DOES EACH PROVIDER HANDLE IT DIFFERENTLY?
----------------------------------------------
Each AI provider invented their own mechanism:

  Anthropic (Claude) — "forced tool use": a synthetic tool named
      "structured_output" is registered, and the model is forced to call it
      exactly once. The tool's input_schema IS your Pydantic schema. The
      call arguments ARE your structured result. No JSON parsing needed.

  OpenAI (GPT-4o) — response_format with json_schema + strict=True.
      The model fills in a JSON object matching your schema.

  Gemini — response_mime_type="application/json" + a response_schema.
      The model returns raw JSON in the content.

HOW DOES THE FRAMEWORK HIDE THIS?
----------------------------------
You define a Pydantic model. You call registry.complete_structured().
The framework picks the right mechanism for the provider automatically.
You always get back a plain dict — parse it into your Pydantic model with
MyModel(**result.data). Same calling code regardless of provider.

Run:
    ANTHROPIC_API_KEY=sk-ant-... python examples/02_structured_output.py
=============================================================================
"""

import asyncio
import os

from pydantic import BaseModel

from mcp_agent_framework import AnthropicClient, Message, ModelRegistry

# ---------------------------------------------------------------------------
# 1. Define the schema for what you want to extract
# ---------------------------------------------------------------------------
# Pydantic BaseModel gives you:
#   - Type hints that become JSON Schema (the framework reads these)
#   - Validation when you construct the object
#   - Clean .field access instead of ["field"] dict access

class ArticleMetadata(BaseModel):
    title: str
    author: str | None           # None if no author is named
    summary: str                 # 1-2 sentence summary of the article
    topics: list[str]            # list of subject tags
    estimated_read_minutes: int  # rough estimate based on word count
    sentiment: str               # "positive", "neutral", or "negative"


# ---------------------------------------------------------------------------
# 2. Sample article to extract from (no real API call needed for the content)
# ---------------------------------------------------------------------------

SAMPLE_ARTICLE = """
Google DeepMind announced AlphaFold 3 this week, extending its protein structure
prediction capabilities to DNA, RNA, and small molecules. Researchers from over
190 countries have already used AlphaFold to accelerate drug discovery. The model
is now available via a web server for non-commercial use. Scientists at the
Wellcome Sanger Institute called it "a new era for structural biology."
"""


# ---------------------------------------------------------------------------
# 3. Extract structured data using the registry
# ---------------------------------------------------------------------------

async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Get your key at https://console.anthropic.com and run:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-..."
        )
        return

    # ModelRegistry lets you register models by name and call them anywhere.
    # Here we register one model; Example 03 shows registering many.
    registry = ModelRegistry()
    registry.register("extractor", AnthropicClient("claude-haiku-4-5-20251001"))

    messages = [
        Message(
            role="user",
            content=f"Extract metadata from this article:\n\n{SAMPLE_ARTICLE}",
        )
    ]

    print("Sending article to Claude and requesting structured extraction...")
    print("-" * 60)

    # complete_structured() handles all the provider-specific machinery.
    # result.data is a plain dict matching the ArticleMetadata schema.
    result = await registry.complete_structured("extractor", messages, ArticleMetadata)

    # Parse the dict into the Pydantic model for typed access.
    # If the model returns unexpected data, Pydantic will raise a ValidationError here.
    article = ArticleMetadata(**result.data)

    print(f"Title:     {article.title}")
    print(f"Author:    {article.author or '(none listed)'}")
    print(f"Topics:    {', '.join(article.topics)}")
    print(f"Summary:   {article.summary}")
    print(f"Sentiment: {article.sentiment}")
    print(f"Read time: {article.estimated_read_minutes} min")
    print("-" * 60)
    print(f"Raw dict from model: {result.data}")
    print(f"\nModel used: {result.model_name}")
    print(
        "\nNotice: you got a typed Python object with guaranteed fields — "
        "no string parsing, no KeyError surprises."
    )


if __name__ == "__main__":
    asyncio.run(main())
