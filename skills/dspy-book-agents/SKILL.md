---
name: dspy-book-agents
description: >-
  Build DSPy agents from chapter 8 of Context Engineering with DSPy — loading
  MCP tools with dspy.Tool.from_mcp_tool and the async contract it forces,
  conversation memory through dspy.History with summarization of older turns,
  Mem0 as retrievable long-term memory, connecting an external vector database,
  and the two ways a multi-hop loop can terminate: agent-controlled with
  variable cost, or a fixed hop budget.
when_to_use: >-
  User says "MCP tools", "conversation history", "the agent forgets",
  "long-running chat", "memory", "multi-hop", "vector database", "Qdrant",
  "agent costs too much"; an agent needs tools, state across turns, or a
  bounded retrieval budget.
---

# Building Agents (book chapter 8)

From the O'Reilly companion repo (MIT). `dspy-retrieval` owns retrieval
mechanics. This skill covers what chapter 8 adds: MCP, memory, and loop budgets.

## MCP tools

The API is `dspy.Tool.from_mcp_tool(session, tool)` — **not** `from_mcp`. DSPy
does not manage the connection: you own the client lifecycle, and the tools are
only valid inside the session's context manager.

```python
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        mcp_tools = await session.list_tools()
        tools = [dspy.Tool.from_mcp_tool(session, t) for t in mcp_tools.tools]
        agent = dspy.ReAct("user_request -> response", tools=tools, max_iters=10)
        return await agent.acall(user_request="...")
```

**The async contract is a hard edge.** A synchronous `agent(...)` call with an
async converted tool raises `ValueError` by default. Use `await agent.acall(...)`.
`dspy.configure(allow_tool_async_sync_conversion=True)` exists but should be a
deliberate choice, not a reflex.

`dspy.Tool.from_langchain(...)` is the sibling adapter, also async.

## Memory, in four escalating steps

**1. `dspy.History`** is a field type, not a store:

```python
history: dspy.History = dspy.InputField()
dspy.History(messages=[{"question": ..., "answer": ...}])
```

Each message dict's **keys must match the signature's other field names** —
not OpenAI's `role`/`content`. Nothing persists; the caller rebuilds and passes
it every turn.

**2. Summarize overflow.** Keep the last N turns verbatim and compress the rest
into a synthetic first turn. The summarizer's instruction is the real work:
*"Summarize older turns without inventing information. Preserve current goals,
constraints, unresolved issues, and corrections. When details conflict, keep
the newest version."*

**3. Store results, not trajectories.** The chapter's own comment: store
`react_result.answer` in history, never `react_result.trajectory`. Otherwise
every large tool payload is replayed on every later call.

**4. Mem0** for retrievable long-term memory, exposed to the agent as three
plain tools wrapping add, search and get-all, wrapped in try/except with a
readiness flag so the agent degrades instead of crashing.

## Retrieval contract

A retriever is any callable returning an object with `.passages`. That is the
entire interface, which is why the same RAG module works unchanged over an
in-memory index and over an external vector database. Swapping the backend
changes one line.

When using an external store, the chapter's warning applies: the collection's
vector dimensions, vector name, embedding model and document field must all
match the client configuration, or retrieval silently returns nothing useful.

## Two ways to end a multi-hop loop

| Approach | Termination | Cost |
|---|---|---|
| agent-controlled `ReAct` | the `finish` tool, or `max_iters` | varies per question |
| fixed-depth module | a `for` loop over `num_hops`, no early exit | predictable |

The fixed-depth version exists for exactly one reason, stated plainly: *"For a
predictable retrieval budget."* Choose deliberately. An agent that decides its
own depth is more capable and less forecastable.

Deduplicate notes and sources between hops, or hop two re-retrieves hop one.

## Cost control is a metric, not a constant

The chapter has **no retry logic, no timeouts except one, and no cost caps**.
What it offers instead:

- `max_iters` on every ReAct, sized to task breadth — 5 for a narrow check, 15 for open research.
- `max_llm_calls` alongside `max_iters` on RLM, an explicit call ceiling distinct from the iteration count.
- A fixed hop budget as the deterministic alternative.
- And the actual answer: a trajectory metric that scores efficiency (full credit for two or fewer tool calls, less beyond) and is handed to an optimizer. **Optimize the agent to be cheap rather than hard-coding a cap.**

Graceful degradation is the error pattern throughout: check for the key, return
a message string rather than raising, keep a readiness flag.

## Framework comparison, reported honestly

The notebook implements the same two-tool agent in DSPy, LangChain, Pydantic
AI, the OpenAI Agents SDK and CrewAI, on one task with one model so the
comparison does not change models between frameworks.

**It contains no conclusion.** There is no verdict cell, no summary table, no
winner. It is a code-shape comparison and the reader draws the conclusion. If
someone cites this notebook as evidence that one framework wins, they are
citing prose that is not in the repository.

## Anti-patterns

- Calling an MCP-tooled agent synchronously; it raises by design.
- Using MCP tools outside the session context manager.
- `dspy.History` messages keyed `role`/`content` instead of by signature field name.
- Replaying trajectories into history, so every turn re-sends old tool payloads.
- A multi-hop loop with no dedup between hops.
- Hard-coding a cost cap instead of optimizing an efficiency metric.
- Quoting the framework comparison's conclusion. There isn't one.

## Where to go next

- Retrieval, recall@k, and the injected-retriever shape → `dspy-retrieval`
- RLM for long context → `dspy-rlm-module`
- Agent tools and code execution → `dspy-book-modules`
- Complete applications built on these parts → `dspy-book-use-cases`
- Full reference (MCP setup, memory code, hop budgets) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_agent_budget.py](example_agent_budget.py)
