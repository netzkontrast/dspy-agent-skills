# Building the Dataset — Reference

Source: chapter 4 (plus chapter 3's dataset step) of
`context-engineering-dspy-book` (MIT). Repo pins `dspy==3.3.0`.

## Conversion recipes

```python
from dspy.datasets import HotPotQA, GSM8K
dataset = HotPotQA(train_seed=1, train_size=100, eval_seed=2024, dev_size=50, test_size=0)
train = [dspy.Example(question=x.question, answer=x.answer).with_inputs('question')
         for x in dataset.train]

# pandas / CSV
df = pd.read_csv('data.csv')
dataset = [dspy.Example(**ex).with_inputs("text") for ex in df.to_dict(orient='records')]
```

`with_inputs` is the contract between the dataset and the program. Fields named
there are given to the program; every other field is gold and stays hidden.

## Splitting

```python
all_examples = list(dataset)
random.Random(2024).shuffle(all_examples)        # seeded literal, not random.seed()
n = len(all_examples)
train = all_examples[:int(0.5 * n)]
val   = all_examples[int(0.5 * n):int(0.75 * n)]
test  = all_examples[int(0.75 * n):]
```

For a large public corpus, the chapter's stated sizes:

| Slice | Size | Note |
|---|---|---|
| train | 200 | deterministically sampled |
| validation | 100 | held out while the optimizer searches |
| test | 500 | final evaluation only |

Validation and test come from the same upstream split but **disjoint index
ranges**. Verify that yourself; nothing enforces it.

## Difficulty tiers

Three tiers, a few examples each, named in the notebook's own comments: clear
cases the baseline should pass, boundary cases it might fail, ambiguous cases
hard for any model. Build the set so every failure mode found in error analysis
appears.

Ambiguous cases go to a separate log with `needs_clarification`,
`default_routing` and `reasoning`. They are not training data; they are a list
of places the task definition is incomplete.

## Error annotation shape

```python
{"ticket": ..., "pred_department": ..., "gold_department": ...,
 "notes": "Model sees 'can't find' and assumes technical issue, "
          "but this is an account management question."}
```

`notes` explains the mechanism, and is the same channel chapter 3 feeds
directly into a GEPA metric as the feedback string:

```python
def exact_match(example, response, trace=None, pred_name=None, pred_trace=None):
    score = 1 if example.is_ai == response.is_ai else 0
    if pred_name:
        return dspy.Prediction(score=score, feedback=example.notes)
    return score
```

The dual-mode return is the technique: a plain number for evaluation, a
`Prediction` with feedback when an optimizer is asking.

## Synthetic generation

```python
class FactGeneration(dspy.Signature):
    """Generate facts and their veracity."""
    sindex: str = dspy.InputField(desc="a random string seed")
    fact: str = dspy.OutputField(desc="a statement about the world")
    veracity: bool = dspy.OutputField(desc="True if fact is correct, False otherwise")

fact_generator = dspy.Predict(FactGeneration, n=15)
response = fact_generator(sindex="seed_001")
examples = [dspy.Example(fact=f, answer=v).with_inputs('fact')
            for f, v in zip(response.completions.fact, response.completions.veracity)]
```

Diversity comes only from `sindex` and `temperature=1`. No filter, no
validation, no collapse check exists in the chapter — add:

- a dedup pass on near-identical generations,
- a quality gate (a metric, or a second model disagreeing),
- a length assertion, because `n>1` is unsupported on several reasoning endpoints and `zip` truncates silently.

## Enrichment (the safer synthesis)

When real outputs exist but the inputs your signature needs do not, write the
inverse signature and infer them:

```python
class TellJoke(dspy.Signature):       # the real task: topic -> joke
class IdentifyTopic(dspy.Signature):  # the inverse: joke -> topic
```

The outputs stay real; only the inputs are synthesized. This is the chapter's
strongest guard against training on a model's own distribution.

## PII synthesis

```python
class SynthesizeTicket(dspy.Signature):
    """Generate a synthetic support ticket with similar structure but fake details."""
    original_ticket: str = dspy.InputField()
    synthetic_ticket: str = dspy.OutputField(desc="Synthetic ticket with fake names, emails, account numbers")
```

Gold label carried over unchanged. Missing from the notebook and required in
practice: a check that the original identifiers are absent from the output, and
an acknowledgement that the real data still reached the provider.

## Known gaps in this chapter

No warning about leakage, contamination or devset overfitting appears anywhere
in chapter 4. Chapter 3 evaluates on the full dataset including train, and one
notebook optimizes against `testset[:20]`. Treat those as defects to avoid, not
patterns to copy, and do not attribute leakage rules to the book.

## Version notes

`dspy.settings.configure(lm=...)` appears in one notebook; `dspy.configure(...)`
is current, and `with dspy.context(lm=teacher):` is the idiomatic way to scope a
teacher pass without mutating global state. `desc=` is the canonical field
keyword; one notebook uses `description=`. Re-compiling an already-compiled
program stacks demos — `deepcopy()` first.
