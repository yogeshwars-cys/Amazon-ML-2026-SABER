# Selector: model and process

The blocker hands over a candidate set with about 99%+ pair recall at 20–30 candidates per S1, and every pair already carries about 30 features (see [BLOCKING.md §7](BLOCKING.md#7-candidate-output-and-pair-features-srcpair_featurespy)). The selector must turn this set into `matching_results.tsv` and maximise **macro F0.5**, which treats precision as twice as important as recall and gives singletons full credit for an empty answer.

The design follows from four data facts:

| Fact (train) | Consequence for the selector |
|---|---|
| Each S2/S3 record matches **at most one** S1 | Candidates compete across S1s: assign each record to at most one S1 |
| 47% of S1 names are shared; 19% of distractors reuse an S1 name | The name alone decides little; address and number agreement, and the competition margin, carry the precision |
| 5.6% of S1 are singletons; each S1 averages about 3.5 matches (1–4 from S2 and 1–4 from S3) | Choose a *set* per S1, including the empty set, and use count priors |
| France is in test but not train; countries are an open set | No country one-hot; use language-agnostic features; stricter cutoffs where there is no supervision |

## Process

```
candidate pairs + features  ──►  (1) pair scorer  ──►  (2) exclusivity / competition
                                                             │
                     (4) group consistency  ◄────────────────┘
                                  │
                     (3) expected-F0.5 set selection per S1  ──►  matching_results.tsv
```

### 1. Pair scorer

Estimates P(match) for each candidate pair.

- **Model:** LightGBM on the blocker's pair features.
  - Leg evidence: cosines, ranks in both directions, leg flags.
  - String similarity: rapidfuzz name and address scores.
  - Structure: house number, number-token Jaccard, region overlap, candidate source.
  - Competition context: candidate counts and within-S1 / within-record ranks.
- **Training data:** train candidates *from the full-density blocker run*, so the negatives have the real difficulty. Train S1s are split by group; the 2.5%-world S1s form the untouched validation set, because they were also held out of encoder training.
- **Calibration:** isotonic regression on a separate fold. Step 3 needs calibrated probabilities, not raw scores.
- **Country:** not used as a feature.

### 2. Exclusivity and competition

Each S2/S3 record belongs to at most one S1:

- For each candidate record, keep only its highest-probability S1, and only if that probability clearly beats the second-best S1.
- Feed the margin `p(best S1) − p(second S1)` back as a feature: a stacked second-round model, or an explicit penalty.
- This targets the false merges between branches of the same chain, which are the most damaging errors under F0.5.

### 3. Expected-F0.5 set selection

For each S1:

1. Sort its surviving candidates by calibrated probability `p_1 ≥ p_2 ≥ …`.
2. For each cutoff `k = 0…n`, estimate the **expected** F0.5 of predicting the top-k. Monte-Carlo sampling of Bernoulli outcomes works, or the exact dynamic program for F-measure-optimal decisions.
   - **Singletons.** The empty set's expected F0.5 is P(no true matches at all). That probability comes from the candidates, `∏(1 − p_j)`, blended with a separate singleton classifier on S1-level features: best-candidate probability, candidate count, name uniqueness.
3. Output the argmax. This optimises the metric directly, per entity, instead of one global threshold, and chooses the empty set when that pays.

### 4. Group consistency

This is the light "semantic graph" step.

- **Mutual agreement.** True S2/S3 matches of one S1 also resemble each other. For each candidate, add its mean similarity to the S1's other high-probability candidates (S2↔S3 agreement) and whether it is the reciprocal best match of those candidates.
- **Count priors.** Add a per-S1 prior over how many matches come from S2 and from S3, learned from train. It damps over-long sets.
- **Where it runs.** Recompute the features after step 2 and run one more scorer round before step 3.

### 5. Optional: cross-encoder reranker

Only if the holdout shows a precision gap after steps 1–4:

- A small Apache or MIT cross-encoder fine-tuned on (S1 text, candidate text) pairs from train.
- Applied only to uncertain pairs (0.2 < p < 0.8) to keep it cheap.
- Base candidates: an arctic-xs or e5-small-sized BERT with a classification head.

## Validation protocol

- **Exact metric.** Compute macro F0.5 per S1, averaged, on the held-out S1s, scored against the *full-density* candidate set.
- **Ceiling vs achieved.** Report both the oracle F0.5 given the blocker's candidates and the achieved F0.5, so blocker losses and selector losses stay separate.
- **Breakdowns.** Report per country and per match-count bucket (0, 1–2, 3–4, 5+), plus the singleton accuracy.
- **France proxy.** Train on one country and test on the other (US → India, India → US) to measure how much the selector degrades on an unseen country. Use that gap to set France's cutoff margin.

## Deliverables it feeds

- **`matching_results.tsv`:** one row per test S1, possibly with an empty list.
- **`candidate_pairs.tsv`:** the blocker's final candidate set, exactly what the selector scores. Every matched id is guaranteed to be in it.
- **Validation:** run `utils/validate_submission.py` before every upload.
