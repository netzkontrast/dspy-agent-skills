# Changelog

## v0.10.0 — 2026-09-16

Merge of two parallel skill lines. Both added skills to the same pack; this release is their union, and the version supersedes the `0.7.0` each claimed independently.

## v0.9.0 — 2026-09-16

### The book, split by chapter

`dspy-context-engineering-book` was one router over 57 notebooks. It is now an index over nine per-chapter skills, each carrying that chapter's transferable technique rather than a table of contents. Every claim was read from the notebook sources, and where a notebook and this pack's verified skills disagree, the disagreement is named.

- `dspy-book-eight-steps` (ch. 1–3) — the build order, and the chapter's distinctive move: optimize the judge before the task, because an unvalidated judge moves the program toward its own errors.
- `dspy-book-datasets` (ch. 4) — conversion recipes, seeded splits, difficulty stratification, and the synthetic-data patterns. Flags that chapter 4 contains **no** leakage or contamination warning and that chapter 3 evaluates on the full dataset including training rows.
- `dspy-book-metrics` (ch. 5) — eleven recipes and the judge-calibration loop, plus a routing table the chapter never states and four gaps it never mentions, including that an optimizer pointed at a judge will exploit that judge.
- `dspy-book-optimizers` (ch. 6) — twelve optimizers measured on one task. Two scored **below** the unoptimized baseline, the two free ones beat six paid ones, and the most expensive run produced the worst result.
- `dspy-book-modules` (ch. 7) — adapters and multimodal, neither covered anywhere else in this pack, plus the rule that DSPy does not select JSONAdapter automatically for a Pydantic output.
- `dspy-book-agents` (ch. 8) — MCP tools and their async contract, conversation memory, and the two ways a multi-hop loop terminates. Records that the framework-comparison notebook has **no conclusion cell**, so any cited verdict is not in the repository.
- `dspy-book-use-cases` (ch. 9) — seven architectures routed by task shape, and the committed-benchmark pattern: assert against a recorded artifact statically, offline and free.
- `dspy-book-production` (ch. 10) — load-once serving, guardrails as optimizable signature fields, typed-only fallback, and the trace-field rule that keeps MLflow's MCP server from exhausting a context window.
- `dspy-book-coding-agents` (ch. 11) — optimizing a SKILL.md or AGENTS.md with `gepa.optimize_anything`, the 20-case adversarial benchmark shape, and the silent trap where renaming the evaluator's `example` parameter drops the dataset.

### Compounding-engineering wiki plan

- Added `docs/compounding-wiki-extension-plan.md`: how to extend the Kohärenz Protokoll wiki with a learnings layer, additively. The finding it rests on is that the wiki already produces learnings in three places (the D-xx decision log, agent memory files, lit-critic triage) and has no path that reads any of them back.
- Every extension point is off by default, mirroring the existing `no_canon_retrieval` pattern, so no current caller changes behaviour.
- The plan also documents what **not** to copy from the upstream, verified in its source: a literal backslash-n that degrades the injected context to one line, a hard-coded similarity of 0.9 that makes its threshold argument inert, a field-name mismatch that leaves auto-codified rows empty and silently disables deduplication on them, a README claiming optimizers that do not exist in the code, an in-place default that edits your current branch, and a missing LICENSE file.

### Validation

- `pytest tests/` passes with the nine new skills
- every new example's `--dry-run` passes with no LM, no network and no service

## v0.8.0 — 2026-09-16

### Six integration skills, and a plugin plan for Kohärenz Protokoll

Each skill teaches one external project, verified against its source rather than its README. Where a README and the code disagreed, the code is documented and the discrepancy is named.

- Added `dspy-refrag` — REFRAG fragment selection. Documents, with file and line evidence, that the fragment selection never reduces the prompt (`forward` joins every passage and only annotates `(selected: bool)`), that `FAISSRetriever` and `PineconeRetriever` raise `NotImplementedError`, that importing the package requires psycopg2, and that the Weaviate pin contradicts the Weaviate code. One file, `sensor_advanced.py`, is worth vendoring.
- Added `dspy-rlm-hooks` — the four RLM lifecycle hooks and speculative execution. Covers the order-dependent composition (hooks before speculation, or speculation silently no-ops), the purity requirement for speculated tools, and the LM-free benchmark harness.
- Added `dspy-drg-kg` — schema-driven knowledge-graph extraction. Leads with the two failure modes that waste the most time: the base install omits DSPy so extraction cannot run, and a missing LM returns an empty graph with only a log warning unless a strict env var is set.
- Added `dspy-tara-rag` — self-corrective RAG. Documents the seven tools (the README says six), the 4D context score, and that progressive leniency terminates by lowering the bar: a context scoring 27 of 100 is accepted at retry 3. Flags that the claimed MIT `LICENSE` file is absent from the repository.
- Added `dspy-tools-cli` — the DSPyTools CLI. Separates the commands that work standalone from those needing FalkorDB, Redis or llama-cpp-server, and reports the source counts where they differ from the README's.
- Added `dspy-context-engineering-book` — a router over the 57 notebooks of the O'Reilly companion repo, mapping each of this pack's optimizer names to its chapter-6 notebook.

### Dependency manifests

- Added `requirements.txt` (dspy, pytest — enough for every dry run and the validators) and `requirements-extras.txt`, grouped by skill. No example requires its extra: each degrades to a deterministic core when the package is absent and asserts the live API surface when present.

### Kohärenz Protokoll plugin plan

- Added `docs/kohaerenz-protokoll-plugin-plan.md`: what to port from each of the six upstreams, with a verdict and a reason per repo, the four concrete integration points in `tools/kpwiki/`, a sequenced plan, and what not to do. Three upstreams are recommended against porting.
- Added `scaffolding/kp_canon_retriever.py` (design note plus scaffolding only, inert): the `CanonRetriever` seam that `SourceIngest` already accepts and currently fills with a no-op, meaning its canon-conflict check never fires.

### A correction to MMR, found by measuring

The scaffold adds a relevance floor to MMR selection, because plain MMR scores an irrelevant passage `0 - 0 = 0` and a relevant near-duplicate slightly below zero — so the irrelevant one wins. Measured on a four-passage fixture, unguarded MMR selected the passage with zero query similarity at every diversity setting from 0.5 to 0.8. The floor fixes it at every setting; the diversity weight alone never does. Both the scaffold and the `dspy-refrag` skill now teach the floor as mandatory.

### Validation

- `pytest tests/` passes with the six new skills
- every new example's `--dry-run` passes, both with and without its package installed
- `dspy-rlm-hooks` and `drg-kg` were installed from source and their asserted API surfaces verified live

## v0.7.0 — 2026-09-16

### Three new skills, consolidated from `OmidZamani/dspy-skills` (MIT)

That pack ships 22 narrow skills; this release ports the parts the pack did not already cover, merged into three skills rather than transplanted one-to-one. Ten source skills map into these three; the rest overlapped `dspy-fundamentals`, `dspy-evaluation-harness` or `dspy-gepa-optimizer` and were left out rather than creating a second source of truth.

- Added `dspy-optimizer-selection` — the whole optimizer family (LabeledFewShot, BootstrapFewShot, BootstrapFewShotWithRandomSearch, KNNFewShot, COPRO, MIPROv2, SIMBA, GEPA, BootstrapFinetune, Ensemble, BetterTogether) as a routing decision: measure a baseline, take the cheapest optimizer whose trainset-size and metric-shape requirements you meet, escalate only on a measured plateau. Absorbs the source pack's six per-optimizer skills. Documents three `compile` signatures that break the common pattern (`Ensemble.compile(programs)`, `KNNFewShot.compile(student, teacher)`, `SIMBA`'s seed on compile) and that `MIPROv2` validates its LMs in the constructor, like GEPA's `reflection_lm`.
- Added `dspy-retrieval` — `dspy.Embedder`, `dspy.Embeddings` (FAISS above 20,000 passages), index persistence, single-hop and deduplicated multi-hop RAG, and the rule that makes RAG debuggable: score recall@k separately from answer accuracy, with a table mapping the four outcomes to the component at fault. Corrects the source pack's global `dspy.configure(rm=...)` + `dspy.Retrieve` pattern to an injected callable retriever, which is testable.
- Added `dspy-production` — cache hardening (`configure_cache(restrict_pickle=True)`), state-JSON versus cloudpickle save formats and which is safe to accept, usage accounting, async `acall`/`aforward`, `streamify` with `StreamListener`, `dspy.Parallel`, plus the observability half: `inspect_history`, `GLOBAL_HISTORY`, the full `BaseCallback` hook list, MLflow autolog, and the sampling/buffering a callback needs to avoid becoming request latency.
- `dspy-advanced-workflow` step 6 now routes through `dspy-optimizer-selection` instead of assuming GEPA, step 7 points at `dspy-production`, and the routing table gained rows for retrieval, optimizer choice and deployment.

### Validation

Every API claim in the three skills was read off the installed wheel with `inspect.signature` rather than copied from prose docs, and each example asserts those signatures so an upstream change fails the smoke test instead of a user's compile run.

- `pytest tests/` -> 249 passed
- 14 of the 15 `skills/*/example_*.py --dry-run` pass, including the three new ones. `skills/dspy-rlm-module/example_rlm.py` still fails in an environment that resolves DSPy 3.3.x, where `dspy.RLM` renamed `max_iterations` to `max_iters`. That failure predates this release and is untouched here: the pack targets the DSPy 3.2.x series, and retargeting the RLM surface is a separate decision.
## v0.7.0 — 2026-09-16

### Three new skills: the wiki loop, the independent judge, the local runtime

- Added `dspy-wiki-compile` — the Karpathy LLM-wiki pattern as a DSPy program, with the details from `llm-wiki-agent`, `llm-wiki-compiler` (two-phase compile, line-range citations), `synthadoc` (decision rules flag / update / create, active-page protection, truncation flag) and `quicky-wiki` (knowledge diff). `BatchCompile` extracts every source before merging any concept, decides `create` in code, and returns drafts only. `compile_metric` is deterministic on five axes (citations resolve, decisions legal, merge, diff consistency, links + language) and names every deficit.
- Added `dspy-adversarial-review` — an independent judge that cannot rewrite, from `synthadoc`'s adversarial gate, AutoSci's `/review` + `/refine`, quicky-wiki's redteam and llm-wiki-compiler's citation-support judge. `assert_independent` refuses a reviewer that is the writer; `Review` carries overstated claims with the evidence they would need; `demotion` changes status, never content; `review_refine` wraps the writer in `dspy.Refine` with the review score as reward; `judge_metric` scores the reviewer's precision and recall so GEPA cannot just make it harsher. Positioned against `dspy-autodialectics` (program-run honesty vs artifact review by a second model).
- Added `dspy-local-runtime` — the `Hmbown/dspy-local` pattern: a `dspy.BaseLM` over `claude -p --output-format json`, backend selection (`api · claude-cli · auto`), the kwargs the CLI cannot honour and what each program loses, a call budget, and GEPA through the CLI. The example's `--probe` makes one real call; verified on DSPy 3.2.1.
- `dspy-advanced-workflow` routing and loop rows; README, `docs/usage.md`, `docs/installation.md`, manifests → 0.7.0.

### Validation

- `pytest tests/` -> passes with the three new skills
- all fifteen `skills/*/example_*.py --dry-run` pass under DSPy 3.2.1; `example_local_runtime.py --probe` answers through the CLI

## v0.6.0 — 2026-09-15

### New skill: `dspy-autodialectics`

- Port of `autodialectics` (an anti-slop agentic harness) into DSPy constructs. Ported: the immutable contract compiler (domain inference, defaults, forbidden shortcuts, sha256), thesis → antithesis → synthesis with typed `Objection`/`Disposition` outputs (replacing the original's four regex parsers), an independent `Verify` predictor that never sees the plan, the 12-dimension `SlopScorer` as a deterministic `dspy.Prediction(score, feedback)` metric, the rubric run score, the accept/revise/reject gate, and the champion/challenger promotion rule with canary cases. Not ported: CLI, REST API, MCP server, CLI gateways, SQLite store, code sandbox (transport and product plumbing).
- The skill is tiered so an agent loads only what it uses: Tier 0 (contract + slop score + gate) needs zero LM calls and ships as importable functions in `example_autodialectics.py`; Tier 1 adds the four-predictor `Dialectic` module; Tier 2 adds GEPA evolution with promotion gated on canaries.
- `dspy-advanced-workflow` gained the routing row and the honesty-gate step in the self-optimizing loop.

### Validation

- `pytest tests/` -> passes with the new skill
- `skills/dspy-autodialectics/example_autodialectics.py --dry-run` passes

## v0.5.0 — 2026-09-15

### New skill: `dspy-tetraframe`

- Compact port of `tetraframe-dspy` (Hmbown, MIT): a decision seed is distilled to one falsifiable predicate; four corners (P, not-P, both, neither) are generated in strict isolation (`CornerView` guard, independent `rollout_id`/temperature per corner, near-duplicate regeneration); contradictions, complementarities, evidence discriminators and invariants are mapped; a non-averaging frame P* is produced with `dspy.BestOfN` under a reward that penalises compromise language.
- Deterministic verification suite with the upstream thresholds (branch independence 0.90, rigor of both/neither 0.78, contradiction honesty 0.75, transformation quality 0.82, fake novelty 0.70, slop 0.70); `tetraframe_metric` returns `dspy.Prediction(score, feedback)` naming every failed check, so the corner generators and the transformer are GEPA-optimizable. `transformation_quality` caps at 1.0 before the compromise penalty (documented deviation) so a compromise P* cannot pass.
- Plugs in before decisions that change authority: wiki merge/supersede/delete proposals from `dspy-deep-refine`, promotion conflicts surfaced by `dspy-clarify`, bundled objectives in `dspy-rlm-workflow`, question → decision records.
- `dspy-advanced-workflow` routing and loop tables updated.

### Validation

- `pytest tests/` -> passes with the new skill
- all eleven `skills/*/example_*.py --dry-run` pass under DSPy 3.2.1

## v0.4.0 — 2026-09-15

### New skill: `dspy-clarify`

- Transposes the `clarify` code skill (Hmbown/clarify, Apache 2.0 — reveal intent, make the implicit explicit, add nothing, never change behaviour) from code to statements: a `ClarifyClaim` Signature returns the same claim with explicit scope (only where the source states it), glossary bindings, labelled assumptions and every remaining ambiguity as a question for the human; verdict `clear` / `needs-author` / `not-promotable`.
- Deterministic `clarify_metric` (meaning kept, scope grounded, hedges resolved, bindings valid, questions well-formed, language kept) so the gate is GEPA-optimizable and cannot be optimized into "sounding precise".
- Plugs in at every authority boundary: research → canon promotion, before decomposition in `dspy-rlm-workflow`, before refining a base for a query in `dspy-deep-refine`, and when a correction enters `dspy-reflect-loop`.
- `dspy-advanced-workflow` routing and loop table updated.

### Validation

- `pytest tests/` -> passes with the new skill
- all ten `skills/*/example_*.py --dry-run` pass under DSPy 3.2.1

## v0.3.0 — 2026-09-15

### Three new skills: the self-optimizing loop

- Added `dspy-rlm-workflow` — port of the `rlm-workflow` skill suite (distill → decompose → solve → synthesize → verify → iterate, after arXiv:2512.24601) as DSPy modules. Decomposition is validated as a DAG in code, the three-tier verification cascade returns `dspy.Prediction(score, feedback)` and doubles as the GEPA metric, runtime iteration uses `dspy.Refine` / `dspy.BestOfN`, distillation uses `dspy.RLM` above ~100k tokens.
- Added `dspy-deep-refine` — port of DeepRefine (arXiv:2605.10488, `DeepRefine-Skill` adapter): judge → k-hop expansion → error abduction on three axes → ≤10 typed refinement actions → deterministic HIGH/MEDIUM/LOW evidence review → approval gate. The module never writes; `apply` refuses LOW by default. A metric (expected triples retrievable after applying non-LOW actions on a copy) makes the refiner GEPA-optimizable.
- Added `dspy-reflect-loop` — port of `claude-reflect-system` v1.3 (signal extraction, review, ledger, promotion, meta-learning): corrections become gold examples plus metric feedback for GEPA instead of templated instruction edits; the fingerprint ledger with a promotion threshold and the accept/modify/skip log become metrics on the reflector itself.
- `dspy-advanced-workflow` gained the routing rows and a "self-optimizing loop" section that orders the four loops (reflect → GEPA → deep-refine → rlm-workflow).
- `scripts/check_dspy_surface.py` now probes `dspy.Refine` and `dspy.BestOfN`.

### Validation

- `pytest tests/` -> passes with the three new skills
- all nine `skills/*/example_*.py --dry-run` pass under DSPy 3.2.1

## v0.2.3 — 2026-05-25

### DSPy 3.2.1 refresh

- Retargeted install and maintainer validation guidance from exact DSPy `3.2.0` to current `3.2.1`, while keeping committed example artifacts labeled by the DSPy version that produced them.
- Added `scripts/check_dspy_surface.py` to validate the live DSPy API surface taught by the skills (`GEPA`, `BetterTogether`, `Evaluate`, `LM`, `SIMBA`, `Embedder`, `configure_cache`, and current primitives).
- Updated GEPA guidance for current upstream best practices: train-heavy GEPA splits, GPT-5-class reflection model shape, literal-dict metric mismatch, supported `component_selector` strings, and when to try `dspy.SIMBA`.
- Tightened the evaluation-harness reference around DSPy 3.2.1 semantics: GEPA-compatible five-argument metric signatures and aggregation-safe metric return shapes.
- Added production cache guidance for `dspy.configure_cache(restrict_pickle=True)`, project-local `DSPY_CACHEDIR`, and provider-side prompt caching.

### Validation

- `uv run --with pytest python -m pytest tests/ -v` -> 114 passed
- `env -u UV_EXCLUDE_NEWER uv run --with dspy==3.2.1 python scripts/check_dspy_surface.py` -> passed

## v0.2.2 — 2026-05-25

### Test suite hardening

- Extended regression guards (`.overall_score`, dict metrics, stale RLM defaults, stale BetterTogether API) to cover `articles/**/*.md` in addition to `skills/` and `docs/`. Uses recursive glob for parity with the `skills/**` pattern.
- Added version consistency test: asserts `plugin.json`, `marketplace.json`, and `README.md` all carry the same version string. Regex is anchored to the `## Version` heading to avoid false matches on changelog or prose mentions.
- Added `reference.md` presence test: every skill directory must ship a `reference.md` for progressive disclosure.
- Moved `_ANTIPATTERN_MARKERS` and `_is_antipattern_context()` above Rule 1 so both Rule 1 (`.overall_score`) and Rule 2 (dict metrics) share the same anti-pattern context check. Added `"enforces"` marker to allow meta-references that describe prohibitions. Dropped the overly broad `"no "` marker — `"enforces"` alone covers the article line that triggered it.
- Test count: 87 → 105.

### Example artifacts

- Re-ran `examples/01-rag-qa` as a clean DSPy 3.1.3 vs 3.2.0 comparison on the same model pair; the current clean DSPy 3.2.0 result is `80.47 -> 100.00`.
- Kept `examples/03-invoice-extraction` on its historical DSPy 3.1.3 artifact after a clean probe: the 3.1.3 GEPA run was stopped before completion after finding a `0.944` candidate, and the 3.2.0 baseline on the same model pair already reached `0.944`.
- Updated README, examples index, and per-example `version_comparison.{md,json}` files so the published docs describe the clean comparison path and no longer depend on `.venv-dspy313` / `.venv-dspy320` state.

### New content

- Created `skills/dspy-advanced-workflow/reference.md` — the only skill that was missing one. Covers step-by-step failure modes, `auto` level selection, plateau debugging, export format tradeoffs, `BetterTogether` chaining, and sub-skill cross-references.
- GEPA constructor snippet marked as a subset with pointer to the full surface in `dspy-gepa-optimizer/reference.md`.
- `reflection_minibatch_size` guidance annotated with symptom context (plateau vs. oscillation) to avoid contradicting the advice in the GEPA optimizer reference.

### Installer

- Added `--verify` flag to `scripts/install.sh`: validates each expected skill exists at the destination, checks symlink targets or directory presence, and reports pass/fail per skill.
- Updated `docs/installation.md` verification section to reference `--verify`.

### Validation

- `uv run --with pytest python -m pytest tests/ -v` -> 108 passed
- Live reruns/probes:
  - `examples/01-rag-qa` -> `80.47 -> 100.00` with `openrouter/mistralai/ministral-3b-2512`
  - `examples/03-invoice-extraction` -> clean probe recorded `0.944` baseline under DSPy 3.2.0; historical artifact retained

## v0.2.1 — 2026-04-28

### Installation

- Documented the supported Vercel `skills` CLI install path: `npx skills add intertwine/dspy-agent-skills`.
- Clarified that current `skills` CLI releases do not resolve the bare package-style source `dspy-agent-skills`; that would require an upstream source alias.
- Fixed `dspy-evaluation-harness` frontmatter so strict YAML parsers used by `npx skills` discover all five skills.
- Added a regression guard for inline YAML frontmatter values containing `: `, which can make strict parsers skip a skill.

## v0.2.0 — 2026-04-21

DSPy 3.2.x refresh for the skill pack. This release candidate moves the skills, references, manifests, and regression guards from DSPy 3.1.x assumptions to the real DSPy 3.2.0 surface, while adding a concrete example for the biggest new optimizer-facing capability.

### Highlights

- Retargeted the repo from DSPy 3.1.x / 3.1.3 to DSPy 3.2.x / 3.2.0 across README, skill docs, manifests, and maintainer guidance.
- Added `skills/dspy-gepa-optimizer/example_bettertogether.py`, a dry-run-capable example of DSPy 3.2.0's generalized `dspy.BetterTogether(metric=..., bootstrap=..., gepa=...)` API.
- Updated `dspy-fundamentals` to document 3.2.x type-mismatch warnings, `warn_on_type_mismatch=False`, and the new `dspy.BaseLM` capability/`ContextWindowExceededError` guidance for custom backends.
- Updated `dspy-rlm-module` for DSPy 3.2.0's `max_output_chars=10_000` default and kwargs-only tool dispatch.
- Updated `dspy-gepa-optimizer` to explain the new BetterTogether chaining model while keeping plain GEPA as the default recommendation.
- Added a regression guard against stale BetterTogether constructor guidance and flipped the RLM default guard to the 3.2.0 value.
- Refreshed `examples/01-rag-qa` and `examples/02-math-reasoning` with clean DSPy 3.2.0 live reruns, and added per-example `version_comparison.{md,json}` files to make the old-vs-new story explicit.
- Kept `examples/03-invoice-extraction` on its historical DSPy 3.1.3 artifact, with the 3.2.0 probe sweep documented instead of forcing a misleading saturated or unstable rerun.
- Validated the install path end to end, including `scripts/install.sh --dry-run`, a temp-`HOME` install, and new guidance for `UV_EXCLUDE_NEWER` when `uv` hides DSPy 3.2.0.

### Validation

- `uv run --with pytest python -m pytest tests/ -v` → full suite passed
- All 6 skill examples executed via `--dry-run` under DSPy 3.2.0
- All 3 end-to-end examples executed via `--dry-run` under DSPy 3.2.0
- Live reruns under DSPy 3.2.0:
  - `examples/01-rag-qa` → `75.77 -> 100.00` with `openrouter/mistralai/ministral-3b-2512`
  - `examples/02-math-reasoning` → `85.00 -> 93.33` with `openrouter/mistralai/ministral-3b-2512`
  - `examples/03-invoice-extraction` → probe sweep recorded saturation or instability; historical artifact retained
- `scripts/install.sh --dry-run` and a temp-`HOME` install both matched the documented dual-target install flow
- During release prep, local `uv run --with dspy` still resolved DSPy `3.1.3` on this machine, so the 3.2.0 smoke tests were run in an isolated environment installed from the official 3.2.0 wheel.

## v0.1.0 — 2026-04-19

First published release. Synthesis and correction of the initial `PLAN.md` draft into a spec-compliant pack that installs cleanly in Claude Code and Codex CLI.

### Skills

- `dspy-fundamentals` — Signatures, Modules, Predict/ChainOfThought/ReAct/ProgramOfThought, save/load
- `dspy-evaluation-harness` — rich-feedback metrics, `dspy.Evaluate`, multi-axis scoring
- `dspy-gepa-optimizer` — full `dspy.GEPA` API (all 22 constructor params)
- `dspy-rlm-module` — `dspy.RLM` long-context / recursive REPL usage
- `dspy-advanced-workflow` — orchestrated end-to-end pipeline

### Corrections to the original PLAN.md draft

| Draft | Correction | Source |
|---|---|---|
| `from dspy.optimizers import GEPA` | `import dspy; dspy.GEPA(...)` (or `from dspy.teleprompt import GEPA`) | https://dspy.ai/api/optimizers/GEPA/overview/ |
| Used `dspy.TypedPredictor` + Pydantic | Use `dspy.Predict` with Pydantic-typed fields (TypedPredictor superseded) | https://dspy.ai/api/modules/Predict/ |
| `dspy.configure(lm=dspy.LM("openai/gpt-5"))` — speculative | Kept `openai/gpt-4o` as default; `DSPY_MODEL` env override | https://dspy.ai/api/models/LM/ |
| GEPA constructor params incomplete (~6 listed) | All 22 params documented with defaults | https://dspy.ai/api/optimizers/GEPA/overview/ |
| `dspy.RLM` args incomplete | Added `max_llm_calls`, `max_output_chars`, `interpreter`; noted Deno requirement | https://dspy.ai/api/modules/RLM/ |
| `dspy.Evaluate(return_all_scores=...)` | `num_threads`, `display_table`, `provide_traceback`, `save_as_csv/json` (the real kwargs) | https://dspy.ai/api/evaluation/Evaluate/ |
| SKILL.md frontmatter used `triggers`, `version`, `dspy-compatibility` | Removed — Claude Code ignores them; version lives in plugin.json. Use `description` + `when_to_use` for auto-invocation | https://code.claude.com/docs/en/skills.md |
| Relied on `npx skillfish add ...` installer | Replaced with official `/plugin marketplace add` path + `scripts/install.sh` for dual-target | https://code.claude.com/docs/en/plugin-marketplaces.md, https://developers.openai.com/codex/skills |
| Single-format distribution | Added `.claude-plugin/{plugin.json, marketplace.json}` + Codex `~/.agents/skills/` support | — |

### Validation / discovered issues

- GEPA asserts `reflection_lm is not None` at **construction time**, not compile — documented as a pitfall in the GEPA skill, and dry-run examples now pass a stub `dspy.LM(...)`.
- 34 tests now cover: SKILL.md frontmatter (spec fields only, kebab-case names, length limits, filename case), plugin/marketplace JSON schemas, and example Python AST parsing.
- All four example scripts execute offline via `--dry-run` against real DSPy 3.1.x.

### Distribution

- Claude Code marketplace manifest (`.claude-plugin/marketplace.json`)
- Claude Code plugin manifest (`.claude-plugin/plugin.json`)
- `scripts/install.sh` for direct install into `~/.claude/skills/` and `~/.agents/skills/` (symlink or copy, idempotent, `--uninstall` supported)

### End-to-end examples

Three validated showcases under `examples/`, each with committed baseline vs. GEPA-optimized numbers:

- `examples/01-rag-qa/` — RAG with citations. GLM 4.5 Air (32B): **81.15 → 100.00 (+18.85)**, 1 mutation accepted.
- `examples/02-math-reasoning/` — multi-step arithmetic. Liquid LFM 2.5 (1.2B): **45.00 → 70.00 (+25.00)**, 5 mutations accepted.
- `examples/03-invoice-extraction/` — Pydantic-typed invoice records. Liquid LFM 2.5 (1.2B): **0.833 → 0.931 (+0.098)**, 5 mutations accepted.

Each example ships `pipeline.py` (module + Signature + metric), `run.py` (CLI: `--dry-run` / `--baseline` / `--optimize` / `--eval`), data JSONL, and `results.{json,md}` from the author's run. Default models are all free on OpenRouter.

### Lessons discovered during validation (baked into the skills)

1. **GEPA metric return must be `dspy.Prediction(score, feedback)`, not a dict.** DSPy's parallel evaluator cannot sum dict-typed outputs (`TypeError: int + dict`). Every example's metric uses the Prediction shape.
2. **GEPA requires `reflection_lm` at construction time**, not at `.compile()`. A cheap stub `dspy.LM(...)` is sufficient for dry-runs (construction is a no-op network-wise).
3. **GEPA state pickling needs cloudpickle** when signatures/modules are dynamic (e.g., typed Pydantic outputs). Pass `gepa_kwargs={"use_cloudpickle": True}`.
4. **`from __future__ import annotations` breaks Pydantic-typed DSPy signatures** — DSPy receives a `ForwardRef` string instead of the actual type. Ex03's pipeline deliberately omits the future import.
5. **`reflection_minibatch_size` matters more than you'd expect.** With small minibatches and high baseline accuracy, GEPA keeps sampling all-correct subsets and the reflection LM is never called. Raise to 6–8 when baseline > 0.7.
6. **Modern 8B+ open models saturate simple extraction and grade-school math**. The math and invoice examples use Liquid 1.2B to create real headroom; stronger models produce baseline ≥ 0.95 and GEPA correctly no-ops.

### GEPA naming correction (v0.1.0 follow-up)

GEPA stands for **Genetic-Pareto**, per the [paper](https://arxiv.org/abs/2507.19457) and every primary source. An earlier version of `skills/dspy-gepa-optimizer/SKILL.md` used the expansion "Genetic-Evolutionary Prompt Adaptation", which turns out to be an LLM-hallucinated backronym with no primary-source support. Fixed the skill; added a short note inline. The articles under `articles/` explain the confusion and where it propagates from.

### Post-release review fixes (Codex audit)

- Replaced `result.overall_score` with the real attribute `result.score` in every skill (`dspy.EvaluationResult` does not have `.overall_score`).
- Skill docs and examples now return `dspy.Prediction(score, feedback)` everywhere — the earlier dict-returning examples would crash `dspy.Evaluate`'s parallel aggregator.
- Added `skills/dspy-rlm-module/example_rlm.py` (every skill now has a runnable `example_*.py`, matching the claim in `docs/usage.md`).
- Fixed `dspy.RLM` `max_output_chars` default: was documented as `10_000`, real default is `100_000`.
- Added `wandb_api_key` and `wandb_init_kwargs` to the GEPA constructor listing.
- Corrected `examples/README.md`'s reproduction instructions (removed references to non-existent `runs/latest/results.json` and `--bench --seeds 5` flag).
- New regression tests (`tests/test_skill_correctness.py`) now fail on: any `.overall_score` in skill docs, dict-returning metrics in skill examples, or a skill missing `example_*.py`.
