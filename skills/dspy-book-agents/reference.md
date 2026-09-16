# Building Agents — Reference

Source: chapter 8 of `context-engineering-dspy-book` (MIT). Repo pins
`dspy==3.3.0`; everything below is current 3.3 surface.

## MCP

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(command=sys.executable, args=[server_script])
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        mcp_tools = await session.list_tools()
        tools = [dspy.Tool.from_mcp_tool(session, t) for t in mcp_tools.tools]
        agent = dspy.ReAct("user_request -> response", tools=tools, max_iters=10)
        result = await agent.acall(user_request="...")
```

The chapter ships its own server over stdio — about 20 lines of `FastMCP` with
`@mcp.tool()` decorated functions and `mcp.run(transport="stdio")`. No port, no
second terminal.

Async rule, verbatim: *"A synchronous `agent(...)` call with an asynchronous
converted tool raises `ValueError` by default. Prefer `await agent.acall(...)`;
enable `dspy.configure(allow_tool_async_sync_conversion=True)` only when you
deliberately want DSPy to bridge the two execution modes."*

Extra installs: `mcp`, and `langchain-core` for `dspy.Tool.from_langchain`.

## Memory

```python
class Chat(dspy.Signature):
    history: dspy.History = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

history = dspy.History(messages=[{"question": "...", "answer": "..."}])
```

Message keys mirror the signature's other field names. First turn passes
`dspy.History(messages=[])`. There is no persistence layer.

Summarization of overflow turns keeps the last N verbatim and re-inserts the
summary as a synthetic first turn. The summarizer signature is
`previous_summary, turns: list[dict[str, str]] -> summary`, and its docstring
carries the real instruction about preserving goals, constraints, unresolved
issues and corrections, newest version winning on conflict.

Mem0: `Memory.from_config(config)` with an LLM and an embedder block, exposed
as three plain ReAct tools around `memory.add`, `memory.search(filters=...)`
and `memory.get_all(filters=...)`, each returning `response.get("results", [])`.

## Retrieval contract

Any callable returning an object with `.passages`. The same RAG module is used
unchanged against an in-memory index and an external vector database:

```python
embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=3)
```

The external-store path uses a third-party package and an older RM-style class,
not core DSPy. Its production warning: the collection's vector dimensions,
vector name, embedding model and document field must all match the client
configuration.

## Multi-hop

```python
# agent-controlled: variable cost
dspy.ReAct("question -> answer, sources: list[str]", tools=[...], max_iters=12)

# fixed budget: predictable cost
for _ in range(self.num_hops):          # default 3, no early exit
    ...
```

Fixed-depth version uses three predictors — `question, notes -> query`,
`question, notes, context -> new_notes: list[str]`, `question, notes -> answer` —
and deduplicates notes and sources with a membership check.

## Reliability and cost

No retries, no try/except around tool bodies, no timeouts except one HTTP call,
no settings-level cost cap. What exists:

| Control | Where |
|---|---|
| `max_iters` | every ReAct; 5 for narrow tasks up to 15 for open research |
| `max_llm_calls` | RLM only; an explicit call ceiling distinct from iterations |
| fixed `num_hops` | the deterministic-budget alternative to ReAct |
| efficiency metric | scores trajectories by tool-call count and is handed to an optimizer |

The efficiency metric is the chapter's real answer to agent cost: full credit
at two or fewer non-terminal tool calls, less beyond, then optimize against it.

Error style throughout is graceful degradation — check the key, return a
message string, set a readiness flag, never raise.

## ReAct details

Tool schema comes from the function name, docstring, argument names and type
hints. Override only when needed with
`dspy.Tool(func=..., name=..., desc=..., arg_desc={...})`.

The trajectory is a **flat dict** keyed `thought_{i}`, `tool_name_{i}`,
`tool_args_{i}`, `observation_{i}` — iterate an index range and break on a
missing key.

Subagents are plain functions that call another ReAct and return a string. The
manager records one tool call; the worker keeps its own trajectory.

The chapter's `calculate` tool uses `eval` and says so: do not expose it to
untrusted input.

## Framework comparison

Five frameworks, one task, one model. The notebook **ends without a conclusion
cell** — no verdict, no table. Its only editorial statements are that the model
is held constant across frameworks and that the async runner is used because
the notebook already owns an event loop. Any claimed conclusion comes from the
book's prose, not this repository.

## Keys and services

Every notebook needs an LLM key. Beyond that: the external vector store needs a
running service and its URL; web search needs a search-API key; the memory
notebook installs `mem0ai` and makes its own model and embedder calls. The
repository states chapter 8 costs roughly one to three dollars end to end.
