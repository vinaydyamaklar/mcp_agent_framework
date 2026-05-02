# Lesson 11 — Evaluation

**Unit 4: Quality and Improvement**

---

## What you will learn

- Why "it looks good to me" is not a production quality signal
- The difference between `LLMEvaluator` and `RubricEvaluator`
- How LLM-as-judge works and when it fails
- How to design criteria weights that match your actual quality bar
- The `EvaluationResult` type and how to act on it

---

## The concept

Your agent produces output. How do you know if it's good?

The naive answer: a human reads it. That works at demo scale. It doesn't work when your agent is running 10,000 evaluations a day, or when you need a machine-readable quality signal to trigger a rewrite loop (Lesson 12).

The framework gives you two evaluators.

---

## `LLMEvaluator` — general-purpose scoring

Uses an LLM to score content on a 0–10 scale, then normalises to 0.0–1.0.

```python
from mcp_agent_framework import LLMEvaluator, AnthropicClient

evaluator = LLMEvaluator(
    llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
    scoring_prompt="""
    Score the following text on technical accuracy and clarity.
    Return a JSON object: {"score": <0-10>, "feedback": "<one sentence reason>"}
    """,
    threshold=0.7,  # scores below 0.7 mark as "not passed"
)

result = await evaluator.evaluate(
    content="Vector search uses cosine similarity to find semantically similar text.",
    context={"task": "Explain vector search in one sentence."},
)

print(result.score)    # e.g. 0.85
print(result.passed)   # True (above threshold)
print(result.feedback) # "Accurate and clear, but could mention embeddings."
```

**The score is clamped to [0.0, 1.0].** Even if the LLM returns 11 or -2 (which happens occasionally), the evaluator applies `max(0.0, min(1.0, raw / 10.0))`.

**When this works well:**
- Qualitative tasks: writing quality, tone, clarity
- "Does this answer the question?" judgments
- Fast evaluation with a cheap model (Haiku)

**When this fails:**
- Factual accuracy (the LLM might not know if the claim is true)
- Math and code correctness (LLMs can't reliably verify computation)
- Domain-specific technical correctness without domain context

For facts and code, use deterministic checkers (regex, unit tests, database lookups) instead.

---

## `RubricEvaluator` — multi-criteria weighted scoring

For when you have explicit, named quality dimensions:

```python
from mcp_agent_framework import RubricEvaluator, RubricCriterion, AnthropicClient

evaluator = RubricEvaluator(
    llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
    criteria=[
        RubricCriterion(
            name="technical_accuracy",
            description="Facts are correct and precise. No misleading statements.",
            weight=0.50,
        ),
        RubricCriterion(
            name="clarity",
            description="A non-expert can understand the explanation without confusion.",
            weight=0.30,
        ),
        RubricCriterion(
            name="conciseness",
            description="No unnecessary repetition or filler. Every sentence adds value.",
            weight=0.20,
        ),
    ],
    threshold=0.75,
)

result = await evaluator.evaluate(content="...")
print(result.score)            # weighted average
print(result.details)          # {"technical_accuracy": 0.9, "clarity": 0.8, "conciseness": 0.7}
print(result.passed)           # True if score >= threshold
```

**All criteria are scored in parallel** using `asyncio.gather`. Three criteria = three simultaneous LLM calls. At N criteria, total time ≈ one LLM call time (not N × one call).

```python
# Inside rubric_evaluator.py:
async def _score_one(criterion: RubricCriterion) -> float:
    # one LLM call to score this criterion
    ...

scores = await asyncio.gather(*[_score_one(c) for c in self._criteria])
```

**Weighted average formula:**
```
final_score = sum(weight_i × score_i for all criteria)
            = 0.50 × 0.90 + 0.30 × 0.80 + 0.20 × 0.70
            = 0.45  +  0.24  +  0.14
            = 0.83
```

---

## `EvaluationResult` — acting on the score

```python
@dataclass
class EvaluationResult:
    score:    float       # 0.0 to 1.0
    passed:   bool        # score >= threshold
    feedback: str         # human-readable reason
    details:  dict        # per-criterion breakdown (RubricEvaluator only)
```

`passed` is the gate. Use it to decide whether to rewrite (Lesson 12) or accept:

```python
result = await evaluator.evaluate(content=draft)
if result.passed:
    return draft
else:
    # trigger rewrite with result.feedback as guidance
    improved = await writer.run(f"Improve this based on: {result.feedback}\n\n{draft}")
```

---

## `AbstractEvaluator` — write your own

Both evaluators implement `AbstractEvaluator`. You can implement your own for domain-specific cases:

```python
from mcp_agent_framework import AbstractEvaluator, EvaluationResult

class CodeEvaluator(AbstractEvaluator):
    async def evaluate(self, content: str, context: dict = None) -> EvaluationResult:
        # Run the code and check if it passes tests
        try:
            exec(content, {})
            return EvaluationResult(score=1.0, passed=True, feedback="Code runs correctly.")
        except Exception as e:
            return EvaluationResult(score=0.0, passed=False, feedback=f"Runtime error: {e}")
```

Any evaluator that implements `evaluate()` works with `EvaluatorOptimizerPattern`.

---

## Designing good rubric criteria

The quality of a rubric determines the quality of the evaluation. Vague criteria produce inconsistent scores.

**Bad criterion:**
```python
RubricCriterion(name="good", description="Is it good?", weight=1.0)
```

**Good criterion:**
```python
RubricCriterion(
    name="actionability",
    description="""
    The response gives the reader clear next steps they can take immediately.
    Score 10: three or more specific, concrete actions with enough detail to execute.
    Score 7: one or two specific actions.
    Score 4: vague suggestions without specifics.
    Score 1: no actionable guidance.
    """,
    weight=0.4,
)
```

Concrete scoring anchors (what a 10, 7, 4, 1 looks like) dramatically improve consistency.

---

## Read these files

```
src/mcp_agent_framework/patterns/evaluation/base_evaluator.py
src/mcp_agent_framework/patterns/evaluation/llm_evaluator.py
src/mcp_agent_framework/patterns/evaluation/rubric_evaluator.py
```

In `rubric_evaluator.py`, find the `asyncio.gather` call and the score clamping. In `llm_evaluator.py`, find how the scoring prompt is used and the 0–10 → 0–1 normalisation.

---

## Run this

```bash
python examples/05_evaluator_optimizer.py
```

Watch the score printed at each rewrite round. Does the score improve? How many rounds until it passes?

---

## Build this

Write a rubric for evaluating customer support email responses:

```python
criteria = [
    RubricCriterion(
        name="empathy",
        description="Response acknowledges the customer's frustration before solving the problem.",
        weight=0.25,
    ),
    RubricCriterion(
        name="solution_completeness",
        description="All aspects of the customer's question are addressed.",
        weight=0.40,
    ),
    RubricCriterion(
        name="professional_tone",
        description="Formal but warm. No slang, no passive-aggressive phrasing.",
        weight=0.20,
    ),
    RubricCriterion(
        name="brevity",
        description="Under 150 words. No unnecessary padding.",
        weight=0.15,
    ),
]
```

Test it on three email responses: one excellent, one mediocre, one bad. Do the scores match your intuition? Adjust the weights and see how the final score changes.

---

## Key terms

| Term | Meaning |
|------|---------|
| `LLMEvaluator` | Single score using one LLM call |
| `RubricEvaluator` | Weighted multi-criteria scoring, criteria evaluated in parallel |
| `EvaluationResult` | Output: score, passed, feedback, details |
| `threshold` | Minimum score to pass evaluation |
| Score clamping | Ensuring score stays in [0.0, 1.0] regardless of LLM output |
| LLM-as-judge | Using an LLM to evaluate LLM output |

---

## Connects to

- **Lesson 12** — `EvaluatorOptimizerPattern` uses evaluators to drive a rewrite loop
- **Lesson 9** — Hierarchy: a parent agent can evaluate a child's output before continuing
- **Lesson 18** — Agentic RAG uses a `check_sufficiency` evaluator to decide whether to search more

---

*Lesson 11 of 21 — Applied AI Engineering*
