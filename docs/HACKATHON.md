# Amazon ML Challenge 2026: business entity resolution

This is a summary of the challenge guidelines as they apply to this project. The official problem statement and portal remain the authority.

## Task

Business records arrive from three independent sources. The records carry no shared identifiers, and their names and addresses are noisy.

- **Source 1 (S1)** is the deduplicated reference source.
- For **each S1 entity**, find every record in **Source 2 (S2)** and **Source 3 (S3)** that refers to the same real-world business.
- An S1 entity can match zero, one, or many S2/S3 records.

## Data

All files are tab-separated. Always read them with an explicit tab separator, because addresses and ID lists contain commas.

Each source file (`*_source1/2/3.tsv`) has four columns:

| Column | Meaning |
|---|---|
| `entity_id` | Record id; the prefix `S1-`, `S2-` or `S3-` gives the source |
| `business_name` | Business name: abbreviations, legal suffixes, typos, transliterations |
| `business_address` | Address: partial, reordered components, landmarks ("Near SBI ATM"), missing PIN or state |
| `country` | Country label. **Open set:** train has US and India, and test adds **France**, which is absent from train |

The ground truth (`train_ground_truth.tsv`) has two columns:
- `source1_entity_id`
- `matched_entity_ids`: comma-separated S2/S3 ids, empty for singletons

The noise to expect:
- **Names:** Corp vs Corporation, Pvt vs Private, Ltd vs Limited, `&` vs "and", DBA/trade names, word transpositions, typos.
- **Addresses:** Rd vs Road, St vs Street, transliteration variants, missing components, landmark references, municipal numbering, component reordering.

## Metric

**Macro F0.5.** It is computed per S1 entity, then averaged over all S1 entities, singletons included.

`F0.5 = 1.25·P·R / (0.25·P + R)`

- Precision counts twice as much as recall.
- A singleton (an S1 with no true matches) scores **1.0** for an empty prediction and **0.0** for any prediction.
- Example: predicting 3 ids when 2 are correct and both true ids are found gives P = 2/3, R = 1, F0.5 = 0.714.

## Deliverables

1. **`matching_results.tsv`** (scored on the leaderboard).
   - Columns: `source1_entity_id`, `matched_entity_ids`.
   - Exactly one row per test S1 entity; `matched_entity_ids` is empty for singletons.
   - No duplicate ids within a list; only S2/S3 ids that exist in the test set.
2. **`candidate_pairs.tsv`** (not scored; used to audit blocking recall and reduction ratio).
   - Columns: `source1_entity_id`, `candidate_entity_ids`.
   - It must be the **final** candidate set that the matching model actually scores.
   - Every matched id must also appear in it.
3. **Final zip:** `output/` (both TSVs), `code/business_entity_resolution/` (`src/`, `README.md`, `requirements.txt`), and the filled-in `Documentation_template.md`. The documentation covers the methodology, blocking strategy, model architecture and features.

A stdlib validator is provided:

```
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Constraints

- **Final model:** MIT or Apache-2.0 licensed, at most **8B parameters**.
- **External data or lookups are strictly prohibited:** no entity-resolution APIs, business registries, geocoding, or internet augmentation. Only the provided training data may be used. Pipelines are audited, and violations mean disqualification.
- **Leaderboard:** the public board uses a subset of test; the final ranking uses the private remainder. Predictions are always submitted for the full test set.

## Data facts measured on train (2026-09-25)

**Sizes**
- Train: 2.2M S1, 5.0M S2 and 5.3M S3 records; 7.64M true pairs.
- Test: 1.73M S1, 4.89M S2 and 5.08M S3 records. About 15% of test S1 records are France.

**Match structure**
- 5.6% of S1 are singletons; S1 records average about 3.5 matches each.
- **Each S2/S3 record matches at most one S1**, and pairs never cross countries.
- About 26% of S2/S3 records are distractors that match nothing.

**Names and scripts**
- 47% of S1 share a normalised name with another S1, and 19% of distractors reuse an S1 name. The address is what separates these hard negatives.
- 13–24% of India S2/S3 names and addresses are in Indic scripts; S1 never is. PIN codes are almost never present.
