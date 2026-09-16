# Metric Recipes — Reference

Source: chapter 5 of `context-engineering-dspy-book` (MIT). The repo pins
`dspy==3.3.0`. Every metric below is the chapter's own, reported as written.

## Return types, and why they matter

`dspy.Evaluate` aggregates with `sum()`, so a metric returning a **dict**
raises `TypeError`. No metric in this chapter does. What they do return:

| Return | Metrics |
|---|---|
| `bool` | `exact_match`, `blog_structure_metric`, `keyword_inclusion_metric` v1, `edit_distance_metric`, `semantic_similarity_metric`, `bleu_metric`, `rouge_metric`, `judge_metric` |
| `float` | `keyword_inclusion_metric` v2, `token_f1`, `marketing_rubric_metric`, `weighted_marketing_metric` |
| `np.float64` | `multi_field_semantic_metric` — cast before JSON logging |
| `dspy.Prediction(score, feedback)` | `llm_judge_metric`, `multi_hop_metric`, `pipeline_metric_with_feedback` |

Only three are GEPA-compatible. The rest use the three-argument form and must
be widened to five before an optimizer can read their feedback.

## String and regex

```python
def exact_match(gold, pred, trace=None):
    return gold.answer == pred.answer
```

Built-ins worth preferring: `dspy.evaluate.metrics.answer_exact_match`
(normalizes punctuation and whitespace) and `answer_passage_match` (does the
answer appear anywhere in `pred.context` — a retrieval check, not an answer
check).

Structure without pinning wording:

```python
header_pattern = r'(?m)^##\s+.+'
return len(re.findall(header_pattern, pred.answer)) >= 3
```

Keywords, the partial-credit version the chapter arrives at:

```python
pattern = r'\b' + re.escape(keyword.lower()) + r'\b'    # \b stops car matching carbon
return matches / len(example.keywords)                   # guard empty: return 1.0
```

Edit distance: `from Levenshtein import distance`, `dist <= threshold`
(default 3). For OCR and spelling only.

## Semantic similarity

```python
embedder = dspy.Embedder("openai/text-embedding-3-small")
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

Threshold guidance: cosine runs −1 to 1; above roughly 0.8 indicates a strong
semantic match; a paraphrase lands near 0.85, unrelated text near 0.45.

Caching helper: key on `hashlib.md5(text.encode()).hexdigest()` into a dict
pickled to disk. Embeddings are deterministic and gold answers are re-embedded
on every candidate prompt during optimization, so the first pass pays and the
rest are free. The chapter defines this helper but never wires it into the
metrics — do that yourself.

`multi_field_semantic_metric` averages similarity across several fields,
skipping empties.

## BLEU, ROUGE, token F1

```python
from nltk.translate.bleu_score import sentence_bleu
sentence_bleu([example.answer.split()], pred.answer.split()) >= 0.5

from rouge import Rouge                      # the `rouge` package, not rouge-score
Rouge().get_scores(pred.answer, example.answer)[0]['rouge-l']['f'] >= 0.5
```

`Rouge()` is instantiated inside the metric body in the notebook, costing per
call. Hoist it.

`token_f1` is set-overlap precision and recall over lowercased whitespace
tokens, with guards for empty sets.

The chapter's verdict, verbatim: *"These traditional NLP metrics tend to
underperform semantic similarity for most DSPy tasks, but they're nearly free
to run and well-validated in the academic literature — worth keeping in your
toolkit, especially when you need a fast sanity check during optimization."*

BLEU's brevity penalty is the chapter's only anti-gaming note: it stops a model
scoring well by emitting very short outputs.

## Judge calibration, in full

```python
class JudgeQuality(dspy.Signature):
    """Evaluate whether a model's answer is accurate and helpful."""
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    gold_answer: str = dspy.InputField(desc="The correct answer for reference")
    is_good: bool = dspy.OutputField(desc="True if answer is accurate and helpful")
    reasoning: str = dspy.OutputField(desc="Explain your judgment")

def judge_metric(example, pred, trace=None):
    return pred.is_good == example.is_good          # plain agreement, no kappa

optimizer = dspy.BootstrapFewShot(metric=judge_metric,
                                  max_bootstrapped_demos=8, max_labeled_demos=8)
split = max(1, int(len(judge_trainset) * 0.8))
optimized_judge = optimizer.compile(student=dspy.ChainOfThought(JudgeQuality),
                                    trainset=judge_trainset[:split])
```

Human labelling runs through `dspy.Evaluate(..., num_threads=1)` — the chapter
marks single-threading CRITICAL, or the interactive prompts interleave. The
human metric writes each label to JSONL as a side effect, so labelling and
trainset construction are one pass. It also asks for reasoning on every label,
not only the verdict.

Agreement is plain label equality. There is no chance correction, so 80% on a
balanced binary task is closer to 60% corrected — treat the chapter's 80–90%
bar as a floor, not a ceiling.

## Rubric

```python
class EvaluateMarketing(dspy.Signature):
    copy: str = dspy.InputField()
    clarity: int = dspy.OutputField(desc="1-5: ...")
    persuasiveness: int = dspy.OutputField(desc="1-5: ...")
    brand_fit: int = dspy.OutputField(desc="1-5: ...")
    emotional_appeal: int = dspy.OutputField(desc="1-5: ...")
    reasoning: str = dspy.OutputField()

weights = {"clarity": 0.3, "persuasiveness": 0.4, "brand_fit": 0.2, "emotional_appeal": 0.1}
normalized = (weighted_sum - 1) / 4.0        # 1-5 -> 0-1
```

Weights must sum to 1.0. Validate by scoring your known-best and known-worst
examples and checking they land where expected.

## Trace-aware metrics

```python
def pipeline_metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
    if trace is None and pred_name is None:
        return is_correct                        # plain evaluation
    _, inputs, outputs = pred_trace[-1]
    if pred_name == "generate_query":
        ...
    elif pred_name == "synthesize":
        ...
    return dspy.Prediction(score=score, feedback=feedback)
```

Iterating `trace` directly yields `(predictor, inputs, outputs)` where the first
element is the predictor **object**, not a name string. Use `pred_name` when you
need the name.

The chapter's own limit: *"Skip them for single-predictor programs or when
evaluation latency matters more than feedback quality — trace analysis adds
overhead on every call."*

## Gaps in the chapter

None of the following appear anywhere in chapter 5. They are this skill's
additions, not the book's:

- An optimizer pointed at an LLM judge will exploit that judge. Re-measure agreement after optimization.
- LLM judges carry position and length bias.
- Repeated optimization against one devset stops measuring quality.
- Cosine similarity cannot detect negation or a reversed fact.

## Do not copy

The chapter's `MultiHopQA` uses `dspy.Retrieve(k=3)` with
`dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")` — the legacy
retrieval path pointed at a public endpoint that is frequently unreachable. Use
`dspy-retrieval`'s injected-callable shape instead.

Model identifiers in this repo (`openai/gpt-5-mini`, `openai/gpt-5.5`) are the
book's choices; verify against what your account actually serves.
