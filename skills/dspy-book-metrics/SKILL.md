---
name: dspy-book-metrics
description: >-
  Pick and write the right DSPy metric for a task, from the eleven worked
  recipes in chapter 5 of Context Engineering with DSPy — exact match, regex
  structure, partial-credit keywords, edit distance, embedding similarity,
  BLEU/ROUGE/token-F1, a calibrated LLM judge, a weighted rubric, and
  trace-aware per-predictor feedback. Carries the chapter's central lesson,
  that binary metrics give an optimizer no gradient, and the judge-calibration
  loop that says when an LLM judge may be trusted as an optimization target.
when_to_use: >-
  User says "how do I score this", "what metric", "my metric is too strict",
  "LLM as judge", "rubric", "the optimizer is not improving", "partial credit",
  "semantic similarity"; a metric must be written before optimizing; a judge's
  trustworthiness is in question.
---

# Metric Recipes (book chapter 5)

From the O'Reilly companion repo (MIT). `dspy-evaluation-harness` owns the
metric *contract* — the five-argument signature and `dspy.Evaluate` mechanics.
This skill owns the *recipes*: which metric shape fits which task, with the
chapter's real implementations.

The chapter is a ladder, each rung added because the previous one broke:
exact match is too strict, regex is all-or-nothing, all-or-nothing is brittle,
edit distance ignores meaning, embeddings need a threshold, subjective tasks
need humans, one judge has blind spots, a rubric still hides *which step*
failed.

## Choose by task shape

The chapter never states this table; it is synthesized from the ladder.

| Your output is | Use | Watch for |
|---|---|---|
| one canonical short answer | `answer_exact_match` (built-in, normalizes) | "The capital is Paris." fails against "Paris" |
| required structure, free wording | regex over the structure | binary; give partial credit instead |
| a set of required points | partial-credit keyword overlap | word-boundary anchors, or `car` matches `carbon` |
| OCR or spelling repair | Levenshtein distance with a threshold | `good`/`great` is 3 edits, `good`/`goof` is 1 |
| a paraphrase of a known answer | embedding cosine | cannot see negation or a flipped fact |
| an extractive span | token F1 | rewards overlap, not correctness |
| a translation or summary | BLEU / ROUGE | the chapter's own verdict: usually weaker than embeddings |
| subjective quality | a judge calibrated against human labels | untrusted below 80% agreement |
| several quality axes | a weighted rubric | weights must be sanity-checked at both extremes |
| a multi-step pipeline | trace-aware per-predictor feedback | adds latency on every call |

## The lesson that matters most

```python
# brittle: one changed word flips 1 -> 0
return all(k in pred.answer.lower() for k in example.keywords)

# usable: the optimizer can see progress
return matches / len(example.keywords)
```

The chapter's words: *"All-or-nothing metrics make optimizers brittle. A small
change can flip the score from 0 to 1 with no smooth gradient in between."*
Every binary metric on the ladder inherits this, including
`semantic_similarity_metric`, which the chapter binarizes at `0.8` without
flagging that it just reintroduced the problem. Return the similarity.

## Calibrating an LLM judge

The chapter's most valuable sequence, and the only one with a stated trust bar.

1. **Label by hand first.** A human metric that prints the case and blocks on input — and always asks *why*, because *"'Why' is more valuable than 'Yes/No' for LLMs."* Run it with `num_threads=1` or the prompts interleave. Each label is appended to a JSONL file, so labelling and trainset construction are one pass.
2. **20 to 50 labels**, more if the task is subtle.
3. **A judge Signature** with `is_good: bool` and `reasoning: str`.
4. **Optimize the judge** against label agreement, plain equality, on 80% of the labels.
5. **Measure on the held-out 20%.** *"If agreement is in the 80–90% range, the judge is trustworthy enough to use as the metric for optimizing your real program. Below that, collect more labels or refine the judge signature."*
6. **Promote it to a GEPA metric**, where the judge's own reasoning becomes the feedback string:

```python
def llm_judge_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    result = optimized_judge(question=example.question, answer=pred.answer,
                             gold_answer=example.answer)
    return dspy.Prediction(score=1.0 if result.is_good else 0.0,
                           feedback=result.reasoning)
```

Do not skip step 5. An uncalibrated judge is an unfalsifiable metric, and
optimizing against one moves the program toward the judge's biases.

## Rubrics

One Signature, several scored fields, normalized to 0–1:

```python
weights = {"clarity": 0.3, "persuasiveness": 0.4, "brand_fit": 0.2, "emotional_appeal": 0.1}
score = sum(getattr(r, k) * w for k, w in weights.items())
normalized = (score - 1) / 4.0          # 1-5 scale -> 0-1
```

Weights must sum to 1.0, and the chapter's check is the right one: score your
known-best and known-worst examples and confirm they land where you expect.

Hoist the judge module out of the metric body. Both chapter notebooks construct
`dspy.ChainOfThought(...)` inside the metric, rebuilding it on every example.

## Trace-aware feedback for pipelines

A final-answer metric cannot say which stage failed. With `pred_name` and
`pred_trace`, the metric addresses one predictor at a time:

```python
def pipeline_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    if trace is None and pred_name is None:
        return is_correct                      # plain eval path
    _, inputs, outputs = pred_trace[-1]
    if pred_name == "generate_query":
        ...                                    # query too short, entities lost
    return dspy.Prediction(score=score, feedback=feedback)
```

Note the dual mode: a plain value when evaluating, a `Prediction` when the
optimizer calls. The chapter's own limit: skip this for single-predictor
programs, because trace analysis costs latency on every call. Also note the
trace tuple's first element is the predictor **object**, not its name.

## What the chapter leaves out

Add these yourself; none of them appear:

- **Judge gaming.** An optimizer pointed at an LLM judge will find the judge's blind spots. Re-measure agreement after optimizing, not only before.
- **Position and length bias** in judges. Longer answers score higher unless controlled.
- **Devset overfitting.** Repeated optimization against one devset stops measuring quality and starts measuring that devset.
- Cosine similarity cannot detect negation. "The treaty was signed" and "The treaty was not signed" sit close together.

## Anti-patterns

- Binarizing a continuous score without asking whether the optimizer needs the gradient.
- Trusting a judge you never measured against human labels.
- Building the judge or an expensive library object inside the metric body.
- Using BLEU or ROUGE as a primary metric for open-ended generation; the chapter positions them as cheap sanity checks.
- Copying the chapter's `MultiHopQA`: it uses the legacy `dspy.Retrieve` plus a hardcoded public ColBERT endpoint that is often unreachable. Use `dspy-retrieval` instead.
- Writing a three-argument metric you will later need for GEPA. Widen it to five now; it costs nothing.

## Where to go next

- The metric contract, devsets and `dspy.Evaluate` → `dspy-evaluation-harness`
- Feeding the feedback string to an optimizer → `dspy-gepa-optimizer`
- Which optimizer to point the metric at → `dspy-optimizer-selection`
- Recall@k, scored separately from answer quality → `dspy-retrieval`
- The notebooks themselves → `dspy-context-engineering-book`
- Full reference (every metric verbatim, return types, caching) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_metric_recipes.py](example_metric_recipes.py)
