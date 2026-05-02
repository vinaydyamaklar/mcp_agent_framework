# Lesson 17 — RAG: Making Agents Know Things

**Unit 6: Knowledge**

---

## What you will learn

- Why RAG exists and what problem it solves
- The index-time and query-time pipelines
- `RecursiveTextChunker` — why chunking strategy matters enormously
- Cosine similarity retrieval via `SemanticMemory`
- Building a complete RAG system end-to-end

---

## The concept

An LLM knows what was in its training data. It doesn't know:
- Your company's internal documentation
- Your customer's order history
- Today's news
- The contents of the PDF you uploaded this morning

RAG (Retrieval-Augmented Generation) solves this: retrieve relevant text at query time and include it in the LLM's context. The LLM answers based on the retrieved text, not just its training data.

```
INDEX TIME (one-time setup):
  Documents → chunk → embed → store in SemanticMemory

QUERY TIME (per request):
  User query → embed → similarity search → top-K chunks → LLM context → answer
```

---

## Index time: chunking

**Why chunk?**

You can't embed an entire document at once. Two problems:
1. **Token limits** — embeddings have input limits (typically 512–8192 tokens)
2. **Dilution** — a 10-page document's embedding averages the meaning of all 10 pages. A query about paragraph 3 won't strongly match the whole-document embedding.

Smaller, focused chunks produce more precise embeddings and better retrieval.

**`RecursiveTextChunker`**

```python
chunker = RecursiveTextChunker(
    chunk_size=500,      # target characters per chunk
    chunk_overlap=50,    # characters shared between adjacent chunks
)
chunks = chunker.split(document_text)
```

"Recursive" means it tries splitting at natural boundaries, largest to smallest:
1. `\n\n` — paragraph breaks (try to split here first)
2. `\n` — line breaks
3. `. ` — sentence ends
4. ` ` — word boundaries (last resort)

It will never split a word. It always tries to keep related sentences together. The overlap (shared characters at chunk boundaries) ensures that a concept split across chunk boundaries isn't lost.

**The chunk size trade-off:**

| Chunk size | Pros | Cons |
|------------|------|------|
| Small (100–200 chars) | Precise retrieval, highly focused | Lost context, fragments ideas |
| Medium (400–600 chars) | Good balance — recommended default | Some context loss at boundaries |
| Large (1000–2000 chars) | Full context preserved | Diluted embeddings, noisy retrieval |

Default: 400–512 characters with 10–15% overlap.

---

## Index time: embedding and storage

```python
from mcp_agent_framework.memory import SemanticMemory

memory = SemanticMemory()

for i, chunk in enumerate(chunks):
    await memory.add(
        chunk,
        metadata={"source": "docs/getting_started.md", "chunk_index": i}
    )
```

Each `add()` call:
1. Embeds the chunk text (converts to a vector)
2. Stores the vector + text + metadata in the memory store

For production, you would plug in a proper embedding model (OpenAI `text-embedding-3-small`, Anthropic's embeddings). For demonstration, the framework uses a TF-IDF-like approach.

---

## Query time: retrieval

```python
results = await memory.search("What is an embedding?", top_k=3)

for r in results:
    print(f"Score: {r.score:.3f}")  # cosine similarity
    print(f"Source: {r.metadata['source']}")
    print(f"Text: {r.content[:100]}...")
```

The `search()` embeds the query and finds the 3 chunks with the highest cosine similarity. Results are sorted highest-to-lowest. Scores above 0.7 are usually relevant; below 0.4 are usually noise.

**The retrieved chunks become context:**

```python
context = "\n\n".join(r.content for r in results)
prompt  = f"Use the following documentation to answer the question.\n\n{context}\n\nQuestion: {query}"
answer  = await llm.complete([Message(role="user", content=prompt)])
```

---

## Wiring it as an MCP tool

The agent calls retrieval as a tool — it doesn't manage the RAG pipeline directly:

```python
rag_app = FastMCP("knowledge_base")
rag_store = SemanticMemory()

@rag_app.tool
async def search_knowledge(query: str) -> str:
    """
    Search the knowledge base for relevant information.
    Use this when you need to answer questions about the documentation.
    Returns the most relevant passages (up to 3).
    """
    results = await rag_store.search(query, top_k=3)
    if not results:
        return "No relevant information found. Try a different search query."
    return "\n\n---\n\n".join(
        f"[Source: {r.metadata.get('source', 'unknown')}]\n{r.content}"
        for r in results
    )

# The agent calls search_knowledge() exactly like any other tool
agent = SingleAgentLoop(
    llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
    config=AgentConfig(mcp_server_config=rag_app),
)
```

---

## The RAG failure modes

**Retrieval failure (retrieved the wrong chunks):**
- Cause: chunks are too large (diluted embeddings), or embedding model doesn't understand domain
- Fix: smaller chunks, better embedding model, or domain fine-tuning

**Context window failure (retrieved too much):**
- Cause: `top_k` too large, chunks too large
- Fix: reduce `top_k`, reduce chunk size, summarise retrieved chunks

**Hallucination despite retrieval:**
- Cause: relevant chunks weren't retrieved (coverage gap in knowledge base)
- Fix: better chunking, more documents, hybrid search (see Lesson 18)

**Query mismatch:**
- Cause: the query words don't match the document words (e.g., "ML model" vs "machine learning algorithm")
- Fix: semantic embeddings handle this better than keyword search; Agentic RAG (Lesson 18) handles it best

---

## Read these files

```
examples/10_rag.py          ← full pipeline: chunker, RAGStore, retrieval, agent
src/mcp_agent_framework/memory/semantic.py  ← add(), search(), cosine similarity
```

In `10_rag.py`, trace the full flow: document text → chunker → memory.add() → search_knowledge tool → agent answer.

---

## Run this

```bash
python examples/10_rag.py
```

Try asking questions whose answers span multiple chunks. Does retrieval find both? Try a question that isn't in the knowledge base — does the agent admit it doesn't know, or does it hallucinate?

---

## Build this

Build a RAG system over your own content. Pick a topic you know well (a README file, a Wikipedia article, a product document). 

1. Load the text
2. Chunk it with `RecursiveTextChunker`
3. Store chunks in `SemanticMemory`
4. Wire as an MCP tool
5. Ask 5 questions: 2 easy (directly in one chunk), 2 medium (requires combining chunks), 1 hard (not in the document)

For each question: did it retrieve the right chunks? Did it answer correctly? Did it admit when it didn't know?

---

## Key terms

| Term | Meaning |
|------|---------|
| RAG | Retrieval-Augmented Generation — augmenting LLM context with retrieved docs |
| Chunking | Splitting documents into smaller pieces for embedding |
| `RecursiveTextChunker` | Splits on natural boundaries: paragraphs → sentences → words |
| `chunk_overlap` | Shared text at chunk boundaries — prevents losing split-boundary context |
| Cosine similarity | How similar two vectors are — 1.0 = identical, 0.0 = unrelated |
| `top_k` | How many chunks to retrieve per query |
| Index time | One-time: chunk → embed → store |
| Query time | Per-request: embed query → search → retrieve → include in context |

---

## Connects to

- **Lesson 7** — `SemanticMemory` is the engine under RAG
- **Lesson 18** — Agentic RAG adds BM25 search, self-evaluation, and multi-round retrieval
- **Lesson 20** — a `research_topic` skill wraps a RAG pipeline as a named capability

---

*Lesson 17 of 21 — Applied AI Engineering*
