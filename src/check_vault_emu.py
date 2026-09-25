"""Check the GPU vault emulation against last session's native-BQBOOST numbers on stock embeddings (work/E.npy):
centred BQBOOST recall@20 was US 0.9484, India 0.835; exact dense US 0.9499, India 0.8361."""
import sys, os, numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(__file__))
from legs import vault_search, dense_search
from bench_encoders import truth_index
from config import WORK as W
E = np.load(W + "E.npy"); E /= np.linalg.norm(E, axis=1, keepdims=True)
q, pool, tp = (pl.read_parquet(W + f) for f in ("q.parquet", "pool.parquet", "tp.parquet")); nq = q.height
for c in ("US", "India"):
    qi, pi, tq, tc = truth_index(q, pool, tp, c)
    Q, P = E[:nq][qi], E[nq:][pi]; mu = np.concatenate([Q, P]).mean(0)
    vi, _ = vault_search(Q, P, 20, shortlist=200, mu=P.mean(0)); di, _ = dense_search(Q.astype(np.float16), P.astype(np.float16), 20)
    print(c, "vault_emu@20", round(float((vi[tq] == tc[:, None]).any(1).mean()), 4),
          "dense@20", round(float((di[tq] == tc[:, None]).any(1).mean()), 4), flush=True)
