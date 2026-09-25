# Blocking: design, experiments and numbers

Blocking caps recall: a true pair the blocker misses can never be matched. SABER therefore aims for **≥ 99% pair recall** while keeping **about 20–30 candidates per S1**, so the selector's work stays affordable. Every number below is measured, and every design choice here traces back to a measurement.

## 1. The evaluation world

All blocker experiments use a scaled-down copy of train, the **2.5% world**:

- **Queries:** 2.5% of S1 per country, 55,085 records.
- **Pool:** every true match of those queries, plus 2.5% of the distractors, 258,015 S2/S3 records.
- **Truth:** 191,064 true pairs.

These records are **held out of encoder fine-tuning**: none of the queries or pool records appears in any training batch, so dense recall on this world is honest.

This pool is about 40× thinner than a real country pool, so the numbers here are optimistic for forward (S1 → S2/S3) search. The full-density run on all of train, scored on the same held-out S1s, is the definitive test.

## 2. Normalisation (`src/norm.py`, `src/prep.py`)

The same function normalises every record:

- **Indic scripts → Latin.** Covers Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and Malayalam via ITRANS, with word-final schwa deletion. For example, `राम मार्केटिंग` becomes `ram marketing`, which lines the S2/S3 Indic records up with Latin S1.
- **Unicode cleanup.** NFKD with combining marks stripped, plus ligatures (œ→oe, æ→ae, ß→ss, ø, ł, đ) for France.
- **Case and punctuation.** Lowercase, `&` becomes "and", and punctuation is dropped unless it sits between alphanumerics.

All 24.2M train and test records are normalised once into parquet, which takes about 18 minutes on 10 processes.

## 3. The encoder: finding a replacement

### 3.1 The stock encoder is the bottleneck

The original vault encoder is all-MiniLM-L6-v2, distilled on SciFact and quantised to W4G16A8, with 192-d Matryoshka output. Its exact dense cosine recall@20 is only US 0.950 / India 0.836. The binary-code search on top of it loses almost nothing: centred sign codes, a 4×K Hamming shortlist and INT8 rescoring give 0.948 / 0.835. So the fix had to be a better encoder, not a better index.

### 3.2 Zero-shot screen (`src/bench_encoders.py`, `results/encoder_screen.jsonl`)

Setup:
- 16 MIT or Apache-2.0 encoders, all at most 8B parameters.
- Input text is `name, address` after normalisation, encoded in fp16 with max length 64.
- Search is exact cosine on the GPU, per country.

| Model | HF id | License | Params (M) | Dim | US@10 | US@20 | US@50 | India@10 | India@20 | India@50 |
|---|---|---|---|---|---|---|---|---|---|---|
| minilm-l6 (stock base) | sentence-transformers/all-MiniLM-L6-v2 | Apache-2.0 | 22.7 | 384 | 0.9314 | 0.9453 | 0.9587 | 0.8067 | 0.8310 | 0.8582 |
| **arctic-xs** | Snowflake/snowflake-arctic-embed-xs | Apache-2.0 | 22.6 | 384 | **0.9854** | **0.9899** | **0.9936** | 0.9130 | 0.9285 | 0.9446 |
| arctic-xs-mean (mean pooling) | Snowflake/snowflake-arctic-embed-xs | Apache-2.0 | 22.7 | 384 | 0.9817 | 0.9868 | 0.9907 | 0.9002 | 0.9161 | 0.9318 |
| granite-30m | ibm-granite/granite-embedding-30m-english | Apache-2.0 | 30.3 | 384 | 0.9785 | 0.9836 | 0.9884 | 0.8812 | 0.8989 | 0.9173 |
| mxbai-xsmall | mixedbread-ai/mxbai-embed-xsmall-v1 | Apache-2.0 | 24.1 | 384 | 0.9497 | 0.9607 | 0.9703 | 0.8345 | 0.8566 | 0.8797 |
| e5-small-v2 | intfloat/e5-small-v2 | MIT | 33.4 | 384 | 0.9818 | 0.9874 | 0.9914 | 0.9192 | 0.9310 | 0.9431 |
| bge-small-en-v1.5 | BAAI/bge-small-en-v1.5 | MIT | 33.4 | 384 | 0.9727 | 0.9799 | 0.9858 | 0.9094 | 0.9266 | 0.9433 |
| gte-small | thenlper/gte-small | MIT | 33.4 | 384 | 0.9735 | 0.9807 | 0.9866 | 0.9047 | 0.9228 | 0.9408 |
| arctic-s | Snowflake/snowflake-arctic-embed-s | Apache-2.0 | 33.2 | 384 | 0.9803 | 0.9863 | 0.9912 | 0.9099 | 0.9264 | 0.9420 |
| minilm-l12 | sentence-transformers/all-MiniLM-L12-v2 | Apache-2.0 | 33.4 | 384 | 0.9216 | 0.9362 | 0.9493 | 0.7781 | 0.8038 | 0.8316 |
| **ml-e5-small** | intfloat/multilingual-e5-small | MIT | 117.7 | 384 | 0.9839 | 0.9884 | 0.9921 | **0.9328** | **0.9462** | **0.9586** |
| para-ml-minilm-l12 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Apache-2.0 | 117.7 | 384 | 0.8464 | 0.8702 | 0.8954 | 0.7036 | 0.7297 | 0.7621 |
| granite-107m-ml | ibm-granite/granite-embedding-107m-multilingual | Apache-2.0 | 107.0 | 384 | 0.9646 | 0.9712 | 0.9769 | 0.8964 | 0.9114 | 0.9270 |
| bge-base-en-v1.5 | BAAI/bge-base-en-v1.5 | MIT | 109.5 | 768 | 0.9763 | 0.9827 | 0.9885 | 0.9096 | 0.9253 | 0.9416 |
| e5-base-v2 | intfloat/e5-base-v2 | MIT | 109.5 | 768 | 0.9784 | 0.9842 | 0.9891 | 0.9223 | 0.9351 | 0.9486 |
| gte-ml-base | Alibaba-NLP/gte-multilingual-base | Apache-2.0 | 305 | 768 | — | — | — | — | — | — |

What the screen shows:
- **Retrieval-tuned small models beat the stock encoder by 4–10 points.** arctic-embed-xs is the best in the US and multilingual-e5-small the best in India.
- **Base-size models do not help.** bge-base and e5-base score no higher than the small ones and encode about 4× slower.
- **Two models were not scored.** gte-multilingual-base's remote code crashed with a CUDA assert under transformers 5.x. bge-m3 (568M) was skipped because encoding 24M records with it would take over 10 hours on a 6 GB laptop GPU.
- **arctic-xs fits the native engine's shape exactly** (6 layers, 384 hidden, 12 heads, 1536 FFN, BERT-uncased 30,522-token vocab). Only its pooling differs: CLS, where the native engine uses mean pooling. The mean-pooled variant is therefore also in the fine-tuning screen; if it matches, the tailored model runs in the native W4 engine unchanged.
- **Throughput.** With the CPU free, 6-layer 384-d models encode about 5,800 records/s in fp16 on the RTX 3050, so 24M records take about 70 minutes. The 12-layer models are roughly 2× slower.

### 3.3 Contrastive fine-tuning (`src/train_encoder.py`, `results/finetune_results.jsonl`)

**Training pairs**
- **Anchor:** an S1 record's `name, address`.
- **Positive:** one of its S2/S3 matches, resampled every time.
- **Data:** 2.03M anchors and 7.45M pairs after removing the 2.5%-world holdout.

**Loss**
- Symmetric InfoNCE (anchor → candidates and positive → anchor) with scale 30.
- Matryoshka over {192, full dim}, so the 192-d prefix is what the 24-byte vault codes carry.

**Hard negatives, built into batch construction**
- Half of the batches are contiguous windows over the **(country, name)-sorted mix of S1 anchors and unmatched S2/S3 distractors**.
- The other half are random batches, which keep the global geometry intact.
- A sorted window looks like the example below: the same business name at different addresses. This is exactly the hard case, since 47% of S1 names are shared. The distractors in the window act as extra negatives.

```
surgical clinic inc, 1119 joffre place beavercreek oh  ||  surgical-clinic, 119 joffre pl dayton oh
surgical clinic inc, 60 quarry street mount vernon ky  ||  surgical clinic inc council, 62 quarry st m vernon kentucky
surgical clinic inc, 1896 greenway avenue columbus oh  ||  surgicalclinic.com, 1896 greenway avenue columbus ohio
NEG: surgical clinic inc, 1476 mani street utah layton
```

**Optimiser:** AdamW at lr 5e-5 with warm-up and cosine decay, fp16 autocast, batch 256. The GPU peaks at 3.2 GB; batch 384 spills out of the 6 GB card and runs 3× slower.

**Result.** arctic-xs, 3,000 steps (20 minutes, about 38% of an epoch):

| arctic-xs | US@10 | US@20 | US@50 | India@10 | India@20 | India@50 |
|---|---|---|---|---|---|---|
| zero-shot, 384-d | 0.9854 | 0.9899 | 0.9936 | 0.9130 | 0.9285 | 0.9446 |
| **fine-tuned, 192-d prefix** | 0.9940 | **0.9964** | 0.9980 | 0.9874 | **0.9924** | 0.9958 |
| fine-tuned, 384-d | 0.9961 | 0.9978 | 0.9988 | 0.9905 | 0.9942 | 0.9967 |

In India, recall@20 goes from 0.835 (stock) to 0.929 (zero-shot) to **0.992**. The dense leg alone now beats the trigram leg (0.975). Screening runs for the mean-pooled arctic-xs, e5-small-v2 and multilingual-e5-small are in progress. The winner then gets a longer run.

## 4. Search legs (`src/legs.py`)

### 4.1 Vault leg: the BQBOOST path on the GPU

This leg reproduces the ranking of the author's BQBOOST vault engine; the engine itself is not included here. The steps:

1. **Centred sign codes.** `bit = (x − μ) > 0`, where μ is the per-country mean embedding.
2. **Hamming shortlist.** Computed as a ±1 fp16 matmul (`dot = d − 2·hamming`), which is exact.
3. **INT8 rescore.** Symmetric per-row INT8 (`scale = max|x| / 127`, round half to even); the score is the int32 dot product times both scales, computed exactly in fp32 on the gathered shortlist.

**Validation (`src/check_vault_emu.py`).** On stock embeddings with the native engine's 200-candidate shortlist, the emulation gives recall@20 US **0.9484** and India **0.835**. That is identical to the native engine run.

**Cost.** Key signs sit on the GPU; the INT8 rows are pinned in host memory and gathered per query block. A full-country search is minutes.

### 4.2 Trigram leg

This is `char_wb` 3-gram TF-IDF (sublinear tf, min_df 2), with cosine computed as sparse keys times dense query blocks on the GPU.

**Performance bug found.** `torch.sparse.mm(K, Q.T)` with a *transposed view* of the query block runs **35× slower**: 1.197 s against 0.034 s per block, on 5.0M key non-zeros × 512 queries. The fix is a contiguous (V × b) query block; that path reaches about 76 GFLOP/s even with the GPU shared.

**Pruning common trigrams (`src/tfidf_maxdf.py`, 2.5% world, recall@K):**

| max_df | nnz/row (US) | US@20 | US@50 | India@20 | India@50 |
|---|---|---|---|---|---|
| 1.0 | 48.7 | 0.9958 | 0.9976 | 0.9746 | 0.9836 |
| 0.05 | 32.6 | 0.9944 | 0.9967 | 0.9731 | 0.9825 |
| 0.02 | 19.4 | 0.9882 | 0.9925 | 0.9602 | 0.9727 |
| 0.01 | 12.5 | 0.9709 | 0.9812 | 0.9426 | 0.9586 |
| 0.005 | 7.8 | 0.9117 | 0.9424 | 0.9029 | 0.9276 |

Pruning is only safe down to about 0.05, and even then brute force over a full country costs O(queries × key non-zeros). For the US (1.32M S1 against 6.19M S2/S3) that is hours per direction. **Region partitioning** (section 5) fixes this.

### 4.3 Key legs

These catch exact duplicates cheaply. R-side buckets with more than 30 records are dropped.

- **Address key:** exact normalised address of at least 12 characters.
- **Name key:** core name (legal suffixes stripped) plus the last two address tokens.

### 4.4 Both directions

Every leg runs S1 → S2/S3 (top 50 raw) **and** S2/S3 → S1 (top 5 raw). Because **each S2/S3 record matches at most one S1**, the reverse direction is very strong and cheap. On the 2.5% world, reverse-trigram top-5 alone recovers **0.986** of India pairs and **0.984** of US pairs.

## 5. Region partitioning (`src/region.py`)

The goal is to search the trigram leg only between records that share a region, with nothing hard-coded, so France works too.

**v1: one region per record** (the last comma part of the address). This lost **4.1%** of true pairs across regions; in a 300k-pair sample, 11.6k of the 12.6k lost pairs were in India. India's address components are shuffled, so the "last part" is often a city (mumbai, thane, new delhi). States also appear under aliases (`UP`), in Indic script (`हरियाणा`, `తెలంగాణ`), or under old names (Andhra Pradesh vs Telangana).

**v2: multi-membership**, which is what the blocker uses.

*Vocabulary, per country*
- Every normalised comma part that appears in at least 0.05% of that country's S1 records.
- S1 is the clean reference source, so this works unsupervised for unseen countries.
- Size: US 401 terms, India 453 terms.

*Aliases, learned from train pairs*
- A frequent non-vocabulary S2/S3 part maps to an S1 region when the mapping holds at least 90% of the time over at least 30 pairs.
- Examples: `texas` → `tx`, `up` → `uttar pradesh`, transliterated state names.
- Count: US 7,455 aliases, India 1,124.

*Membership*
- A record belongs to every region named by any address component (the whole part, its first or last token, or its first or last two tokens), directly or through an alias.
- **Two records are compared if they share any region, or if either has none.** Records with no region are searched against the whole country.

| | S1 unknown | S2/S3 unknown | Regions per record | S1 → S2/S3 comparisons vs full | True pairs kept |
|---|---|---|---|---|---|
| US | 0.0% | 3.6% | 1.65 | **15%** | 98.68% |
| India | 0.0% | 3.0% | 2.70 | **22%** | 99.94% |

The 1.3% of US pairs lost to partitioning is still covered by the vault leg, which always searches the whole country.

## 6. Hybrid blocker on the 2.5% world

`src/blocker.py` runs in two phases:

1. **search:** runs every leg per country and caches the raw lists.
2. **merge:** applies adaptive K, takes the union, computes pair features, and writes the TSV.

**Adaptive K**, applied to each list:
- Keep rank < kmin, **or** score ≥ top-1 score − gap, up to kmax.
- Defaults: forward vault 5 / 30 / 0.10; forward trigram 5 / 30 / 0.15; reverse lists kmin 2, kmax 5.

This run uses the **stock** encoder; the fine-tuned one is being swapped in:

| Country | Pairs | Candidates per S1 | **Union recall** | Vault fwd | Vault rev | Trigram fwd | Trigram rev | Address key | Name key |
|---|---|---|---|---|---|---|---|---|---|
| US | 816k | 24.8 | **0.9978** | 0.888 | 0.960 | 0.937 | 0.984 | 0.085 | 0.131 |
| India | 643k | 29.1 | **0.9903** | 0.773 | 0.865 | 0.913 | 0.986 | 0.085 | 0.082 |

(Per-leg recalls are measured after pruning.)

For comparison, the previous best union (stock vault + trigram, forward only, top 20) reached US 0.9967 and India 0.9779.

The pruning sweep (`--tune`, `results/block_report_stock_mini.json`) shows:
- Recall is flat across the forward-list gap and kmax settings: most candidates, and most of the recall, come from the reverse lists.
- The reverse lists are the main lever between candidate count and recall. Cutting reverse-trigram kmax from 5 to 1 drops India recall to 0.983 but saves 7 candidates per S1.
- Final tuning must happen at full density, where forward lists get crowded but reverse lists keep the same per-record size.

## 7. Candidate output and pair features (`src/pair_features.py`)

Each country produces a `cand_*.parquet`: one row per (S1, candidate), with `label` on train. `candidate_pairs_<split>.tsv` is written in the submission format. The features:

- **Leg evidence**
  - Exact vault cosine `cos_v` and trigram cosine `cos_t`.
  - Membership flags for each leg, and the rank in every forward and reverse list (99 when absent).
- **Competition context**
  - `n_cand_s1` and `n_cand_r` (how many S1s want this record).
  - The rank of `cos_v` within the S1's candidates and within the record's S1s.
- **String similarity** (rapidfuzz `cpdist`, multithreaded)
  - Name: ratio and token-set.
  - Core name: partial ratio, Jaro-Winkler, equality.
  - Address: token-set, token-sort, partial.
  - Name length difference.
- **Structure**
  - Number-token Jaccard and house-number agreement.
  - Empty candidate address, region overlap, candidate source (S2/S3).

Class separation (India, 2.5% world, stock encoder):

| label | pairs | name token-set | address token-set | cos_v | cos_t | house_eq | region_overlap | num_jacc |
|---|---|---|---|---|---|---|---|---|
| non-match | 566,958 | 54.6 | 52.8 | 0.649 | 0.258 | −0.13 | 0.71 | 0.015 |
| match | 76,097 | 86.9 | 90.3 | 0.850 | 0.796 | 0.50 | 0.93 | 0.642 |

## 8. Next and prospects

1. **Pick the encoder.** Finish the screening fine-tunes (mean-pooled arctic-xs for native-engine compatibility, e5-small-v2, multilingual-e5-small for France) and give the winner a longer run.
   - Planned: about 2 epochs, plus a second round of hard negatives mined with the fine-tuned model itself.
2. **Full-density run on train.**
   - Encode all 12.5M train records.
   - Run all legs per country, scoring recall on the held-out S1s against the full pool.
   - Tune adaptive K for ≥ 99.5% recall within the candidate budget.
3. **Test run.** Produce `candidate_pairs.tsv` for 1.73M S1 across US, India and France.
   - France has no training labels. Its region vocabulary is built from test S1, and the trigram and key legs are language-agnostic, which gives a safety net against encoder drift.
4. **Native export (optional).** If mean-pooled arctic-xs wins, run W4G16A8 QAT on the fine-tuned weights so the native vault engine can serve the tailored encoder unchanged.
5. **Separate name and address vectors (optional).** Separate name and address codes would help the selector break same-name ties.
