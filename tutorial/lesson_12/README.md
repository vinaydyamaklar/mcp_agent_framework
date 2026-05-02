# Lesson 12 — EvaluatorOptimizer

**Unit 4: Quality and Improvement**

---

## What you will learn

- The generate → evaluate → rewrite loop
- The critical bug that breaks this pattern (and the exact fix)
- How to set a convergence threshold and max rounds
- When self-improvement is worth the cost
- Real-world use cases where this pattern shines

---

## The concept

A single LLM call produces "pretty good" output most of the time. But for high-stakes tasks — technical reports, marketing copy, production code — "pretty good" isn't good enough.

`EvaluatorOptimizerPattern` implements a self-improvement loop:

```
user_message → LLM → draft
                       ↓
                  evaluator → score
                       ↓
             score < threshold?
             YES → LLM rewrite (with evaluator feedback as guidance)
             NO  → return draft  ← converged
```

After N rounds (or when the score passes the threshold), the best draft is returned.

---

## The most important implementation detail

This is the bug that breaks almost every first implementation of this pattern.

Anthropic's API (and most LLM APIs) requires **alternating roles** in the message history: user → assistant → user → assistant. Two consecutive `role="user"` messages cause a 400 error.

Naive implementation (broken):

```python
# Round 1:
messages = [Message(role="user", content="Write me a report on...")]
draft = await llm.run(messages)   # appends assistant message
# Round 2 (BUG):
messages.append(Message(role="user", content=f"Your score was 0.6. Rewrite it."))
# Now we have: user → assistant → user → user  ← INVALID
messages.append(Message(role="user", content="Write me a report on..."))  # next run also adds user!
```

The fix is to separate the original task from the rewrite instruction:

```python
original_task = user_message
current_prompt = user_message

for round in range(max_rounds):
    draft = await llm.run(current_prompt, history=working_history)

    result = await evaluator.evaluate(draft)
    if result.passed:
        return draft

    # Append this round's exchange to history
    working_history.append(Message(role="assistant", content=draft))

    # The NEXT prompt (not a new history entry) carries the feedback
    current_prompt = (
        f"Your previous response scored {result.score:.2f}/1.0. "
        f"Feedback: {result.feedback}\n\n"
        f"Rewrite to address this feedback. Original task: {original_task}"
    )
    # On the next iteration, run(current_prompt, history=working_history) adds:
    #   history[-1] = assistant (the draft above)
    #   new user message = current_prompt (the feedback)
    # Result: assistant → user → assistant → user → assistant ✓ (valid alternation)
```

The key insight: **`current_prompt` becomes the next `user` message, not a new history entry.** History ends with `assistant` (the last draft). The new user message is the feedback. Valid alternation maintained.

---

## Configuration

```python
from mcp_agent_framework import EvaluatorOptimizerPattern, LLMEvaluator, RubricEvaluator

pattern = EvaluatorOptimizerPattern(
    generator_client=AnthropicClient("claude-sonnet-4-6"),    # writes drafts
    evaluator=RubricEvaluator(
        llm_client=AnthropicClient("claude-haiku-4-5-20251001"),  # scores
        criteria=[...],
        threshold=0.85,
    ),
    config=AgentConfig(mcp_server_config=app, system_prompt="..."),
    max_rounds=4,    # max rewrite attempts before returning best-so-far
)

result = await pattern.run("Write a technical blog post about vector databases.")
```

**Two-model setup is the standard cost optimisation:**
- `generator_client`: capable model (Sonnet, GPT-4o) — writes the actual content
- `evaluator.llm_client`: cheap model (Haiku, GPT-4o-mini) — just scores and explains

The evaluator runs more frequently (once per round) and the scoring task is simpler than generation — so a cheap model works well here.

---

## What happens after `max_rounds`

If the evaluator never passes after `max_rounds`, the pattern returns the **best draft so far** (highest score), not the last draft. This is the correct behaviour — the last draft isn't necessarily better than an earlier one.

---

## When self-improvement is worth the cost

**Worth it when:**
- Output quality has a measurable, well-defined bar (rubric criteria)
- The task is high-stakes (customer-facing content, documentation, reports)
- The generator model produces inconsistent quality on the first try
- You can afford 2–4× the cost of a single generation

**Not worth it when:**
- Speed is more important than quality
- The task is simple enough that the first try is usually good enough
- You don't have a clear quality bar (can't write good rubric criteria)
- The task requires factual accuracy — LLM evaluators won't catch factual errors

---

## Read this file

```
src/mcp_agent_framework/patterns/evaluator_optimizer_pattern.py
```

Focus on:
- How `working_history` and `current_prompt` are managed across rounds
- Where `original_task` is preserved
- The best-score tracking for the `max_rounds` fallback

---

## Run this

```bash
python examples/05_evaluator_optimizer.py
```

Watch the score at each round. Try changing the `threshold` to 0.95 (harder to pass) — does it take more rounds? Try changing the rubric criteria and observe how the rewrites shift focus.

---

## Build this

Build an "email quality improver." The agent rewrites a draft email until it passes a quality bar:

```python
rubric = RubricEvaluator(
    llm_client=AnthropicClient("claude-haiku-4-5-20251001"),
    criteria=[
        RubricCriterion("professional_tone",   weight=0.4, description="Formal, no slang, respectful."),
        RubricCriterion("clarity",              weight=0.3, description="Recipient immediately understands the ask."),
        RubricCriterion("conciseness",          weight=0.3, description="Under 100 words. No filler phrases."),
    ],
    threshold=0.85,
)

pattern = EvaluatorOptimizerPattern(
    generator_client=AnthropicClient("claude-haiku-4-5-20251001"),
    evaluator=rubric,
    config=AgentConfig(mcp_server_config=FastMCP("empty")),
    max_rounds=3,
)
```

Start with a deliberately poor draft:
```
hey!! just wanted to check in about that thing we talked about last time.
let me know asap cause its kinda urgent lol. thanks!!
```

Watch the rewrite rounds. Print the draft and score at each round. Does it converge before hitting `max_rounds`?

---

## Key terms

| Term | Meaning |
|------|---------|
| Generator | The LLM that writes drafts |
| Evaluator | The evaluator that scores drafts |
| Round | One generate + evaluate cycle |
| Threshold | Score that counts as "good enough" |
| `current_prompt` | The user message for the next round — carries feedback |
| `working_history` | Accumulated conversation ending in assistant messages |
| Alternating roles | user → assistant → user → assistant — required by most APIs |

---

## Connects to

- **Lesson 11** — evaluation: the evaluator in this pattern uses `LLMEvaluator` or `RubricEvaluator`
- **Lesson 13** — PlannerExecutor can apply evaluation to each plan step
- **Lesson 9** — Hierarchy: a parent can run an EvaluatorOptimizer loop on a child's output

---

*Lesson 12 of 20 — Applied AI Engineering*
