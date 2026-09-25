"""Trigram leg on the 2.5% world: recall and GPU time vs max_df pruning of common trigrams."""
import json, time, numpy as np, polars as pl, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from legs import tfidf_search
from bench_encoders import truth_index
from config import WORK as W
T = json.load(open(W + "texts.json")); q, pool, tp = (pl.read_parquet(W + f) for f in ("q.parquet", "pool.parquet", "tp.parquet"))
for c in ("US", "India"):
    qi, pi, tq, tc = truth_index(q, pool, tp, c)
    for mdf in (1.0, 0.05, 0.02, 0.01, 0.005):
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, min_df=2, max_df=mdf, dtype=np.float32)
        P = v.fit_transform([T["pt"][j] for j in pi]); Q = v.transform([T["qt"][j] for j in qi])
        torch.cuda.synchronize(); t = time.time(); top, _ = tfidf_search(Q, P, 50); torch.cuda.synchronize(); dt = time.time() - t
        r = {K: round(float((top[tq, :K] == tc[:, None]).any(1).mean()), 4) for K in (10, 20, 50)}
        print(c, "max_df", mdf, "nnz/row", round(P.nnz / P.shape[0], 1), "recall", r, f"{dt:.1f}s",
              f"ns per (query x key nnz) {dt / (Q.shape[0] * P.nnz) * 1e9:.4f}", flush=True)
