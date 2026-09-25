"""Zero-shot encoder screen on the 2.5% scaled train world (work/q.parquet, pool.parquet, tp.parquet, texts.json).
Dense exact cosine recall@K per country on GPU, plus encode throughput. Appends one JSON line per model."""
import sys, json, time, gc, numpy as np, polars as pl, torch
from sentence_transformers import SentenceTransformer

from config import WORK as W
KS = (10, 20, 50)
# (hf id, license, prefix, extra kwargs)
MODELS = {
    "minilm-l6": ("sentence-transformers/all-MiniLM-L6-v2", "Apache-2.0", "", {}),
    "minilm-l12": ("sentence-transformers/all-MiniLM-L12-v2", "Apache-2.0", "", {}),
    "para-ml-minilm-l12": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "Apache-2.0", "", {}),
    "e5-small-v2": ("intfloat/e5-small-v2", "MIT", "query: ", {}),
    "ml-e5-small": ("intfloat/multilingual-e5-small", "MIT", "query: ", {}),
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", "MIT", "", {}),
    "gte-small": ("thenlper/gte-small", "MIT", "", {}),
    "arctic-xs": ("Snowflake/snowflake-arctic-embed-xs", "Apache-2.0", "", {}),
    "arctic-xs-mean": ("Snowflake/snowflake-arctic-embed-xs", "Apache-2.0", "", {"pool": "mean"}),
    "arctic-s": ("Snowflake/snowflake-arctic-embed-s", "Apache-2.0", "", {}),
    "granite-30m": ("ibm-granite/granite-embedding-30m-english", "Apache-2.0", "", {}),
    "granite-107m-ml": ("ibm-granite/granite-embedding-107m-multilingual", "Apache-2.0", "", {}),
    "mxbai-xsmall": ("mixedbread-ai/mxbai-embed-xsmall-v1", "Apache-2.0", "", {}),
    "bge-base-en-v1.5": ("BAAI/bge-base-en-v1.5", "MIT", "", {}),
    "e5-base-v2": ("intfloat/e5-base-v2", "MIT", "query: ", {}),
    "gte-ml-base": ("Alibaba-NLP/gte-multilingual-base", "Apache-2.0", "", {"trust_remote_code": True}),
    "bge-m3": ("BAAI/bge-m3", "MIT", "", {}),
}


def load_model(name, init=None, fp16=False):
    """SentenceTransformer for MODELS[name]; kw 'pool' overrides pooling (mean = native vault engine pooling)."""
    from sentence_transformers import models
    hf, lic, prefix, kw = MODELS[name]; kw = dict(kw); pool = kw.pop("pool", None)
    mk = {"torch_dtype": torch.float16} if fp16 else {}
    if pool and not init:
        tr = models.Transformer(hf, max_seq_length=64, model_args=mk)
        return SentenceTransformer(modules=[tr, models.Pooling(tr.get_word_embedding_dimension(), pool)], device="cuda")
    return SentenceTransformer(init or hf, device="cuda", model_kwargs=mk, **kw)


def truth_index(q, pool, tp, country):
    qi = np.where((q["country"] == country).to_numpy())[0]; pi = np.where((pool["country"] == country).to_numpy())[0]
    qpos = {e: j for j, e in enumerate(q["entity_id"].to_numpy()[qi])}
    ppos = {e: j for j, e in enumerate(pool["entity_id"].to_numpy()[pi])}
    t = tp.filter(pl.col("source1_entity_id").is_in(list(qpos)))
    return qi, pi, np.array([qpos[a] for a in t["source1_entity_id"]]), np.array([ppos[b] for b in t["matched_entity_ids"]])


def topk_dense(Q, P, k, dev="cuda"):
    Pt = torch.from_numpy(P).to(dev, torch.float16); out = []
    for i in range(0, len(Q), 4096):
        s = torch.from_numpy(Q[i:i + 4096]).to(dev, torch.float16) @ Pt.T
        out.append(s.topk(k, dim=1).indices.cpu().numpy())
    return np.concatenate(out)


def recall(E, nq, q, pool, tp):
    Eq, Ep = E[:nq], E[nq:]; res = {}
    for c in ("US", "India"):
        qi, pi, tq, tc = truth_index(q, pool, tp, c)
        top = topk_dense(Eq[qi], Ep[pi], max(KS))
        for K in KS:
            res[f"{c}@{K}"] = round(float((top[tq, :K] == tc[:, None]).any(1).mean()), 4)
    return res


if __name__ == "__main__":
    T = json.load(open(W + "texts.json")); texts = T["qt"] + T["pt"]; nq = len(T["qt"])
    q, pool, tp = (pl.read_parquet(W + f) for f in ("q.parquet", "pool.parquet", "tp.parquet"))
    for name in sys.argv[1:]:
        hf, lic, prefix, kw = MODELS[name]
        try:
            m = load_model(name, fp16=True)
            m.max_seq_length = 64
            nparam = sum(p.numel() for p in m.parameters())
            t = time.time()
            E = m.encode([prefix + x for x in texts], batch_size=512, normalize_embeddings=True, convert_to_numpy=True)
            dt = time.time() - t
            r = recall(E.astype(np.float32), nq, q, pool, tp)
            rec = {"model": name, "hf": hf, "license": lic, "params_M": round(nparam / 1e6, 1), "dim": E.shape[1],
                   "rec_per_s": round(len(texts) / dt), **r}
        except Exception as e:
            rec = {"model": name, "hf": hf, "error": repr(e)[:300]}
        print(json.dumps(rec), flush=True)
        with open(W + "encoder_screen.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
        m = None; gc.collect(); torch.cuda.empty_cache()
