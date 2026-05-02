# Lesson 18 — Agentic RAG

**Unit 6: Knowledge**

---

## What you will learn

- Why standard RAG fails on complex questions
- BM25 keyword search — the math and why it complements vector search
- The `check_sufficiency` self-evaluation loop
- Multi-round retrieval: the agent decides when it has enough
- Hybrid search combining semantic + keyword retrieval

---

## The concept

Standard RAG (Lesson 17) is passive: one query → cosine search → top-K chunks → answer. The agent has no control over retrieval. If the initial chunks are bad, the answer is bad.

**Agentic RAG gives the agent control:**

```
Agent receives question
    ↓
Agent plans which queries to run
    ↓
Agent runs semantic search AND keyword search
    ↓
Agent calls check_sufficiency() → "do I have enough to answer well?"
    ↓
NOT SUFFICIENT → agent re-searches with different queries
SUFFICIENT     → agent synthesises answer
```

The agent iterates until it's confident it has the right information, or until `max_iterations` is hit.

---

## BM25 — keyword search

Semantic search (cosine similarity) is great at finding *conceptually similar* chunks. BM25 finds chunks that contain the *exact words* from the query. These are complementary:

- Query: "HNSW algorithm" → semantic search finds "approximate nearest neighbour" (related concept). BM25 finds chunks that literally say "HNSW".
- Query: "latency p99" → semantic search might return general performance discussion. BM25 finds chunks with "p99" in them.

**The BM25 formula:**

```
BM25(query, document) = Σ IDF(term) × TF(term, doc) × (k+1) / (TF(term, doc) + k(1 - b + b × dl/avgdl))

Where:
  IDF(term)     = log((N - n_t + 0.5) / (n_t + 0.5))  — inverse document frequency
  TF(term, doc) = count of term in document             — term frequency
  dl            = document length
  avgdl         = average document length across corpus
  k = 1.5, b = 0.75  — tuning parameters (standard defaults)
```

You don't need to memorise this. The key intuitions:
- **IDF:** rare words are more informative than common words. "HNSW" (rare) has higher IDF than "search" (common).
- **TF normalised:** raw term count, but with diminishing returns (k parameter) and length normalisation (b parameter). A document isn't twice as relevant just because it's twice as long.

---

## The `check_sufficiency` tool

This is the key innovation. The agent calls this tool to evaluate whether its current retrieved context is complete:

```python
@app.tool
async def check_sufficiency(question: str, context: str) -> str:
    """
    Evaluate whether the retrieved context is sufficient to answer the question.
    Returns: SUFFICIENT or NOT_SUFFICIENT with a reason.
    """
    evaluator_prompt = f"""
    Question: {question}
    
    Retrieved context:
    {context}
    
    Can you answer the question completely and accurately from this context alone?
    If anything crucial is missing, say what it is.
    Respond with either:
    SUFFICIENT: [brief reason]
    NOT_SUFFICIENT: [what is missing]
    """
    resp = await llm.complete([Message(role="user", content=evaluator_prompt)])
    return resp.content
```

When the agent gets `NOT_SUFFICIENT: missing information about chunk size defaults`, it knows to run another search specifically for "chunk size defaults" — a much more targeted query than the original.

This is the self-evaluation pattern from Lesson 11 applied to retrieval rather than output quality.

---

## The full Agentic RAG tool set

```python
agentic_rag_app = FastMCP("agentic_rag")
rag_store = AgenticRAGStore()  # dual-mode: semantic + BM25

@agentic_rag_app.tool
async def search_semantic(query: str, top_k: int = 3) -> str:
    """Search by semantic meaning. Best for conceptual questions."""
    results = await rag_store.search_semantic(query, top_k=top_k)
    return format_results(results)

@agentic_rag_app.tool
async def search_keyword(query: str, top_k: int = 3) -> str:
    """Search by exact keywords. Best for specific terms, names, or technical identifiers."""
    results = await rag_store.search_keyword(query, top_k=top_k)
    return format_results(results)

@agentic_rag_app.tool
async def check_sufficiency(question: str, context: str) -> str:
    """Evaluate if the retrieved context is sufficient to answer the question."""
    ...

@agentic_rag_app.tool
async def get_document_stats() -> str:
    """Get statistics about the knowledge base: document count, topics covered."""
    return rag_store.get_stats()
```

The system prompt guides the agent through the OBSERVE → PLAN → SEARCH → EVALUATE → SYNTHESISE loop:

```python
system_prompt = """
You are a research assistant with access to a knowledge base.

Process:
1. OBSERVE: Call get_document_stats() to understand what's available
2. PLAN: Decide which search strategies to use (semantic, keyword, or both)
3. SEARCH: Run your searches
4. EVALUATE: Call check_sufficiency() with your question and all retrieved context
5. If NOT_SUFFICIENT: re-search with more specific queries (up to 3 rounds)
6. SYNTHESISE: Write a complete answer citing specific sources

Always use both search tools for important questions — they find different things.
"""
```

---

## Hybrid search result merging

After running both searches, you have two ranked lists. How do you merge them?

**Reciprocal Rank Fusion (RRF):**

```python
def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

# Merge two ranked lists
scores = {}
for rank, result in enumerate(semantic_results):
    scores[result.id] = scores.get(result.id, 0) + rrf_score(rank)
for rank, result in enumerate(keyword_results):
    scores[result.id] = scores.get(result.id, 0) + rrf_score(rank)

merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

A result that ranks highly in both lists gets a combined high score. A result that only appears in one list gets only its rank contribution. RRF is robust to scale differences between semantic scores (0–1) and BM25 scores (0–∞).

---

## Why agentic > naive RAG

| | Naive RAG | Agentic RAG |
|--|-----------|-------------|
| Query strategy | Fixed: one query | Dynamic: agent decides |
| Retrieval rounds | 1 | Up to N (self-evaluated) |
| Search type | Semantic only | Semantic + keyword (hybrid) |
| Coverage | Whatever the first query finds | Agent seeks out gaps |
| Cost | 1 retrieval + 1 generation | N retrievals + N evaluations + 1 generation |
| Recall on complex questions | Low | High |

The cost is higher — but for high-stakes questions where "I don't know" or a wrong answer is expensive, the improved recall is worth it.

---

## Read this file

```
examples/11_agentic_rag.py
```

Trace through:
- `AgenticRAGStore`: dual-mode search
- `search_keyword()`: BM25 implementation (find the IDF and TF calculations)
- `check_sufficiency` tool: self-evaluation loop
- The system prompt: OBSERVE → PLAN → SEARCH → EVALUATE → SYNTHESISE

---

## Run this

```bash
python examples/11_agentic_rag.py
```

Ask a question that requires combining two pieces of information from different documents. Watch the search rounds — does the agent issue a second search? What does `check_sufficiency` return on the first round?

---

## Build this

Compare naive vs agentic RAG on the same knowledge base from Lesson 17:

1. Build both: a plain `search_knowledge` tool (naive) and the full `AgenticRAGStore` with `check_sufficiency` (agentic)
2. Ask these 5 questions:
   - "What is BM25?" (easy — directly answerable)
   - "How does BM25 compare to vector search?" (medium — requires both)
   - "What chunk size should I use and why?" (medium — specific technical detail)
   - "How do I build a production RAG system end-to-end?" (hard — requires many pieces)
   - "What is the capital of France?" (not in KB — should admit it doesn't know)

For each question: which approach gets it right? How many rounds does Agentic RAG take? Which questions specifically benefit from multi-round retrieval?

---

## Key terms

| Term | Meaning |
|------|---------|
| BM25 | Probabilistic keyword ranking: TF × IDF with length normalisation |
| IDF | Inverse Document Frequency — rare words score higher |
| TF | Term Frequency — how often the search term appears in a document |
| `check_sufficiency` | Self-evaluation: "do I have enough context to answer?" |
| Multi-round retrieval | Agent searches again if context is insufficient |
| Hybrid search | Combining semantic (cosine) + keyword (BM25) results |
| RRF | Reciprocal Rank Fusion — clean way to merge two ranked lists |

---

## Connects to

- **Lesson 17** — RAG: the foundation this builds on
- **Lesson 11** — evaluation: `check_sufficiency` uses the LLM-as-judge pattern
- **Lesson 20** — a `research_topic` skill wraps Agentic RAG as a named capability

---

*Lesson 18 of 21 — Applied AI Engineering*
