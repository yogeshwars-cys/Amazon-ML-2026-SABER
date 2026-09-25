# Amazon ML 2026 — SABER

**SABER** is a pipeline for the Amazon ML Challenge 2026 business entity resolution task. It works in two stages:

1. **Blocking.** A hybrid, recall-first candidate generator. Its main leg is a sentence encoder fine-tuned for this task, searched through binary sign codes with INT8 rescoring. A char-trigram leg split by region and exact-key legs are added to it, and every leg runs in both directions.
2. **Selection** (next). A precision-first model picks, for each Source-1 business, the subset of candidates that maximises the expected macro F0.5.

This repository documents the blocking work so far, the measurements behind each design decision, and the plan for the selector.

| Document | Contents |
|---|---|
| [docs/HACKATHON.md](docs/HACKATHON.md) | Challenge rules, data format, metric, and deliverables |
| [docs/BLOCKING.md](docs/BLOCKING.md) | Full blocking design, experiments, numbers, and dead ends |
| [docs/SELECTOR.md](docs/SELECTOR.md) | Selector model and process, built on the blocker's outputs |
| [results/](results/) | Raw result files: encoder screen, fine-tunes, and blocker reports |

## Status (2026-09-25)

| Step | State |
|---|---|
| EDA and data facts | done |
| Normalisation (Indic → Latin transliteration, NFKD, ligatures), 24.2M records | done |
| Zero-shot encoder screen, 16 MIT/Apache candidates | done |
| Contrastive fine-tuning, first screening run (arctic-embed-xs) | done: **India recall@20 0.835 → 0.992** |
| Other screening fine-tunes (mean-pooled arctic-xs, e5-small-v2, multilingual-e5-small) | running |
| Hybrid blocker: vault, trigram, key legs, adaptive K, both directions, pair features | done, validated end to end on the 2.5% world |
| Full-density blocker run on all of train (holdout-scored), then test `candidate_pairs.tsv` | next |
| Selector | planned: [docs/SELECTOR.md](docs/SELECTOR.md) |

## Headline numbers

The **2.5% world** is 55k held-out S1 queries against a pool of 258k S2/S3 records, with 191k true pairs. None of these records are used in encoder training. Recall is the share of true (S1, S2/S3) pairs that land in the top K candidates.

| Recall@20, dense leg only | US | India |
|---|---|---|
| Stock vault encoder (SciFact-distilled W4 MiniLM-L6, 192-d) | 0.948 | 0.835 |
| Best zero-shot replacement: Snowflake arctic-embed-xs | 0.990 | 0.929 |
| **arctic-embed-xs fine-tuned, 3k steps, 192-d vault prefix** | **0.9964** | **0.9924** |
| Char-trigram TF-IDF, for reference | 0.996 | 0.975 |

The full hybrid blocker with the *stock* encoder and default adaptive K already reaches **US 0.9978 at 24.8 candidates per S1** and **India 0.9903 at 29.1 candidates per S1** ([details](docs/BLOCKING.md#6-hybrid-blocker-on-the-25-world)). The fine-tuned encoder is being swapped in for the full-density run.

## Layout

```
src/
  config.py          paths (SABER_DATA, SABER_WORK environment variables)
  norm.py            text normalisation: Indic transliteration, NFKD, ligatures, punctuation
  prep.py            normalise every train/test record once -> parquet, plus the exploded train pairs
  bench_encoders.py  zero-shot encoder screen (dense recall@K per country, throughput)
  train_encoder.py   contrastive fine-tuning (symmetric InfoNCE, Matryoshka 192/full, name-sorted hard batches)
  encode.py          bulk GPU encoding (fp16) of a split with a (fine-tuned) encoder
  legs.py            GPU search legs: vault (sign-code Hamming shortlist + INT8 rescore), exact dense,
                     trigram TF-IDF (chunked sparse x dense), region-partitioned trigram
  region.py          data-driven region membership (no hard-coded countries or states)
  blocker.py         search per country -> cached legs -> adaptive-K merge -> candidates + features + TSV
  pair_features.py   rapidfuzz / structural pair features for the selector
  tfidf_maxdf.py     experiment: trigram recall and speed vs max_df pruning
  check_vault_emu.py experiment: GPU vault emulation vs native engine numbers
results/             encoder_screen.jsonl, finetune_results.jsonl, block_report_*.json
```

## Reproduce

```bash
export SABER_DATA=/path/to/student_resource/dataset    # train/ and test/ TSVs from the challenge
export SABER_WORK=/path/to/scratch                     # ~40 GB free space for the full run
pip install -r requirements.txt

python src/prep.py                                     # normalise all records (~18 min on 10 cores)
python src/bench_encoders.py arctic-xs e5-small-v2     # zero-shot screen (needs the 2.5% world files, see docs)
python src/train_encoder.py arctic-xs --steps 3000 --batch 256 --tag _s3k
python src/encode.py --model $SABER_WORK/ft_arctic-xs_s3k --tag ftxs --split train
python src/blocker.py search --split train --emb ftxs
python src/blocker.py merge  --split train --emb ftxs  # recall report + candidates + features + TSV
```

Hardware used: 16 GB RAM, RTX 3050 laptop GPU (6 GB), 12 CPU threads, on Windows.

## Rules this project follows

- **Models:** only MIT or Apache-2.0 models of at most 8B parameters. Every encoder here has at most 118M parameters.
- **Data:** no external data or lookups. Training uses only the provided train split; the region aliases are learned from train pairs.
- **Countries are an open set.** The test set adds France, which is absent from train. Every stage runs per country, and the region vocabulary for an unseen country is learned unsupervised from that country's Source-1 records.
- **Not in this repo:** the challenge data, embeddings, and candidate files.
