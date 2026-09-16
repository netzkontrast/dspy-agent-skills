# The Eight Steps — Reference

Source: chapters 1 to 3 of `context-engineering-dspy-book` (MIT). Repo pins
`dspy==3.3.0`.

## The steps, as the notebook names them

1. Specify your signatures
2. Build your modules
3. Explore a few examples
4. Collect your dataset
5. Define your metrics
6. Establish a baseline
7. Optimize your program
8. Test and iterate

The worked example defines two signatures — a task and its judge — because the
metric is itself a program that must be optimized.

## Step 4

```python
df = pd.read_csv('../data/ai_vs_human.csv')
dataset = [dspy.Example(**ex).with_inputs("text") for ex in df.to_dict(orient='records')]
random.seed(42)
random.shuffle(dataset)
trainset, valset = dataset[:len(dataset)//2], dataset[len(dataset)//2:]
```

The CSV carries a `notes` column. That column becomes the optimizer's feedback
string in step 5 — the dataset ships the diagnosis, not just the label.

## Step 5: the dual-mode metric

```python
def exact_match(example, response, trace=None, pred_name=None, pred_trace=None):
    score = 1 if example.is_ai == response.is_ai else 0
    if pred_name:
        return dspy.Prediction(score=score, feedback=example.notes)
    return score
```

`pred_name` is only set when an optimizer is asking. The same function serves
`dspy.Evaluate` and GEPA.

The task metric is a judge-as-metric: run the detector on the transformed text
and reward fooling it, with feedback assembled from the verdict, the judge's
reasoning, a similar labelled example and its notes.

## Step 7: judge first, then task

```python
judge_optimizer = dspy.GEPA(metric=exact_match, max_full_evals=3, num_threads=4,
                            track_stats=True, use_merge=False, reflection_lm=smart_lm)
optimized_judge = judge_optimizer.compile(AIDetector(), trainset=trainset, valset=valset)
optimized_judge.save("./ai_detector/", save_program=True)
loaded = dspy.load("./ai_detector/")
```

Then a closure rebinds the task metric to the optimized judge, and a second
GEPA run optimizes the transformer against it.

## Step 8

```python
def _best_of_n_reward(kwargs, pred):            # NOT the metric signature
    dummy = dspy.Example(**kwargs).with_inputs(*kwargs.keys())
    return float(optimized_llm_judge(dummy, pred))

selector = dspy.BestOfN(module=optimized_transformer, N=5,
                        reward_fn=_best_of_n_reward, threshold=1.0)
```

The calling-convention mismatch is real and documented: `BestOfN.reward_fn`
receives a kwargs dict and a prediction, not the five-argument metric list.

Portability check: `optimized_transformer.set_lm(dspy.LM(<other provider>))`,
then re-evaluate. A program that only works on one model is overfitted to it.

## The deterministic alternative

The quickstart notebook replaces the judge with a table of roughly eighteen
compiled regex patterns for known tells (em dashes, curly quotes, and a list of
overused words). It is strict — any single hit scores zero — and its feedback
names the matched patterns and the hit count. The dataset is ten or so
hand-written pairs, each commented with the tell it exemplifies, and the final
steps probe one sentence per tell to confirm it is gone.

Free, non-drifting, and more specific than a judge. Prefer it when the quality
you want can be written as a pattern.

## Defects not to inherit

- The evaluation in chapter 3 scores on the **full dataset**, training rows included.
- The quickstart optimizes against a slice of its own test set.
- Field keyword usage is inconsistent (`description=` in one chapter, `desc=` in another); `desc` is canonical.
- `Evaluate.__call__` returns an `EvaluationResult`; always take `.score`.
- Model identifiers are forward-dated placeholders and do not resolve.
