# DSPyTools CLI — Reference

Source: `dspytools` 0.1.0, read from `src/dspytools/` at
`github.com/netzkontrast/dspytools`. Counts below were taken from source, not
from the README.

Entry point: `[project.scripts] dspytools = "dspytools.main:cli"`. Python
`>= 3.12`. Commands are registered in `main.py` via
`cli.add_lazy_command(name, module_path, attr_name)`; there is no
`commands/__init__.py`.

## Command groups

**Core:** `configure` (6 nested subgroups: keys, LM, adapter, cache, dspy,
completion), `signature`, `module`, `run`, `compile`, `evaluate`, `data`,
`doctor` (a bare command, not a group).

**Long tail:** `agent`, `tool`, `inspect`, `mcp`, `server`, `self`, `gfl`,
`skills`, `generate`, `pipeline`, `export`, `compare`, `lora`, `distill`,
`graph`, `memory`.

## Commands with verified flags

```
dspytools doctor [--check-llm/--no-llm] [--check-gpu/--no-gpu]
                 [--check-config/--no-config] [--check-services/--no-services]
dspytools configure key set PROVIDER [KEY] [--stdin]
dspytools configure key list
dspytools configure lm set MODEL [--api-base URL] [--provider P]
                              [--role student|teacher|default]
dspytools signature new "in: str = desc -> out: str" [--name/-n N] [--model M]
                                                     [--instructions/-i TXT]
dspytools module new NAME [--signature/-s SIG] [--model M]
        [--type/-t Predict|ChainOfThought|ReActV2|MultiChainComparison|Tool|Generate|Flex]
        [--instructions/-i TXT] [--from-prompt "in -> out"]
dspytools run predict "question -> answer" [-i KEY=VALUE]... [--lm M]
        [--temperature 0.2] [--max-tokens 4096] [--adapter chat|json|xml|baml]
dspytools run cot   <same flags as predict>
dspytools run react "sig" [-i K=V] [-t TOOL|server:tool]... [--max-iters 10]
dspytools data load SOURCE [--format huggingface|json|csv|auto] [--split train]
        [--name N] [--fields a,b] [--rename old=new] [--input-keys a,b]
        [--limit N] [--output/-o F] [--raw]
dspytools compile <optimizer> MODULE_NAME TRAINSET_PATH [--label L] [--force]
dspytools compile gepa MODULE TRAINSET [--use-mlflow/--no-use-mlflow]
                                       [--gepa-kwargs k=v]...
dspytools evaluate run MODULE DEVSET [--metric exact_match|passage_match|semantic_f1]
        [--num-threads 1] [--display-table] [--lm M]
dspytools skills find QUERY [--k 10] [--category/-c C]
dspytools skills discover [--category/-c C] [--k 20]
dspytools graph status
dspytools self optimize [--teacher/--no-teacher] [--force/-f]
dspytools self evolve [--question/-q Q] [--check]
```

## Subsystems

| Package | Purpose | Key names |
|---|---|---|
| `skills/` | BM25 + embedding search over local and user skill directories | `SkillLoader`, `SkillManager`, `search_external`, `popular_skills` |
| `graph/` | **FalkorDB** (not Neo4j) skill graph, semantic and MCP caches | `GraphClient`, `get_graph_client`, `FalkorDBSkillGraph`, `SemanticCache`, `RedisCache` |
| `evolve/` | self-improvement loop and routing | `SelfEvolveEngine`, `get_engine`, `RouterAgent`, `MorphologyTracker`, `UCBExplorer` |
| `generators/` | DSPy modules that emit Python source for signatures and modules | `SignatureGeneratorDSPy`, `ModuleGeneratorDSPy` |
| `gfl/` | Generate → Evaluate → Keep → Learn → Deploy | `GFLPipeline(mode="compare"\|"single")`, `GFLLoop`, `LSETracker` |
| `mcp/` | MCP server and tool loading | `create_mcp_server`, `run_stdio`, `run_sse(host, port=8002)`, `load_mcp_tools_sync` |
| `memory/` | FalkorDB-native memory | `MemoryManager`, `get_memory_manager` |
| `api/` | FastAPI hot-swap inference server | `server.py` |

`evolve/metrics.py` is a pure re-export shim over `core/metrics.py`
(`auto_metric`, `gepa_metric`, `simple_metric`, `content_quality_score`). Do
not treat it as an independent metrics system — the repo's SSOT rule means each
concern has exactly one owner.

## Environment variables

None are mandatory. Endpoints: `DSPYTOOLS_LLM_URL` (falls back to the student
`api_base`, then `http://127.0.0.1:8000`), `DSPYTOOLS_EMBEDDING_URL`
(`:8001/v1`), `DSPYTOOLS_EMBEDDING_MODEL` (`embeddinggemma`),
`DSPYTOOLS_EMBEDDING_DIM` (768), `DSPYTOOLS_LLAMA_CPP_URL` (`:8080`),
`MLFLOW_TRACKING_URI` (`http://localhost:5000`).

Paths: `DSPYTOOLS_<NAME>_DIR` for `CONFIG`, `DATA`, `CACHE`, `SKILLS`,
`ADAPTERS`, `PROJECT`, `PROJECT_CONFIG`, `COMPILED`, `SIGNATURES`, `MODULES`,
`AGENTS`, `DISTILL`. Also `DSPYTOOLS_ENV_FILE`, `DSPY_LORA_DIR`.

Keys live in `~/.config/dspytools/.env` as `<PROVIDER>_API_KEY`, managed
through `configure key`.

## Dependencies

`dspy-ai==3.3.1` hard-pinned **and** `dspy>=3.3.0` listed alongside it, which is
redundant and a likely source of resolution conflicts. Also click, rich-click,
fastapi, starlette, uvicorn, `mcp>=1.28.1`, `mlflow[mcp]>=3.5.1`, a
pre-release `pydantic>=2.14.0a1`, `numpy>=2.5`, falkordb, redis, redisvl,
optuna, datasets, structlog, jinja2, toml, plus ruff, pytest and pre-commit as
**runtime** dependencies.

Services: FalkorDB + Redis via `docker-compose.redis.yml` (port 6379). MLflow
optional on 5000. Mojo is optional — `try_load_mojo()` returns `(False, None)`
when the SDK is absent and the Python fallback is used.

## Count check

| README claim | Source count |
|---|---|
| 24 command groups | 23 groups + 1 standalone command |
| 166 subcommands | 183 leaves (173 decorated + 10 generated) |
| 17+ optimizers | 19 (10 registry + 9 hand-written) |
| 11 arXiv implementations | 10 distinct identifiers; `gfl/paper_optimizers.py` itself says "six patterns", one without an identifier |
| 102 source files | exact |

Four identifiers are dated 2026 and unverifiable offline. Use the CLI freely;
do not cite its paper list as provenance without checking.

## Surfaces to test before relying on them

`compile submit`/`status`/`cancel` imply a job scheduler (`core/scheduler.py`);
`graph cascade`, `gfl spin|opsd|lse|loop` and `self watch`/`auto-fix` are the
least conventional. Exercise them on a throwaway project first.
