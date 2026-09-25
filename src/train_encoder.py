"""Contrastive fine-tuning of a sentence encoder for business-record blocking.

anchor = S1 "name, addr"; positive = one of its S2/S3 matches (resampled each time).
Loss: symmetric InfoNCE over in-batch candidates (+ extra distractor negatives), Matryoshka over (192, full).
Hard negatives come from the batch construction: half of the batches are windows of the (country, name)-sorted
mix of S1 anchors and unmatched S2/S3 distractors, so same-name / near-name businesses compete in one batch.
Holdout: every S1 in work/q.parquet and every S2/S3 in work/pool.parquet (the 2.5% eval world) is excluded.
With --fit-parts E, anchors are restricted to partition E (partitions.py): matcher-partition S1s and the S2/S3 records
they match are excluded too, so the encoder features the selector trains on are out-of-sample, as on test."""
import argparse, json, math, os, sys, time, numpy as np, polars as pl, torch, torch.nn.functional as F
from sentence_transformers import SentenceTransformer
sys.path.insert(0, os.path.dirname(__file__))
from bench_encoders import MODELS, recall, load_model

from config import WORK as W


def txt(df): return (df["name"] + ", " + df["addr"]).to_list()


def load_data(parts=None):
    import pickle
    cache = W + (f"ft_data_{parts}.pkl" if parts else "ft_data.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f: return pickle.load(f)
    d = _load_data(parts)
    with open(cache, "wb") as f: pickle.dump(d, f, protocol=5)
    return d


def _load_data(parts=None):
    s1 = pl.read_parquet(W + "train_source1.parquet")
    rest = pl.concat([pl.read_parquet(W + "train_source2.parquet"), pl.read_parquet(W + "train_source3.parquet")])
    pairs = pl.read_parquet(W + "train_pairs.parquet")
    hq = set(pl.read_parquet(W + "q.parquet")["entity_id"].to_list())
    hp = set(pl.read_parquet(W + "pool.parquet")["entity_id"].to_list())
    if parts:                                                     # drop other partitions' S1s and their matches
        from partitions import load as part_ids
        keep = part_ids(parts); gone = pairs.filter(~pl.col("s1").is_in(keep.implode()))["m"]
        s1 = s1.filter(pl.col("entity_id").is_in(keep.implode())); hp |= set(gone.to_list())
    s1 = s1.filter(~pl.col("entity_id").is_in(list(hq)))
    rest = rest.filter(~pl.col("entity_id").is_in(list(hp)))
    pairs = pairs.filter(pl.col("s1").is_in(s1["entity_id"].implode()) & pl.col("m").is_in(rest["entity_id"].implode()))
    rid = {e: i for i, e in enumerate(rest["entity_id"].to_list())}
    sid = {e: i for i, e in enumerate(s1["entity_id"].to_list())}
    a = np.fromiter((sid[x] for x in pairs["s1"].to_list()), np.int64, pairs.height)
    b = np.fromiter((rid[x] for x in pairs["m"].to_list()), np.int64, pairs.height)
    o = np.argsort(a, kind="stable"); a, b = a[o], b[o]
    start = np.searchsorted(a, np.arange(s1.height + 1))           # CSR: positives of S1 i are b[start[i]:start[i+1]]
    has = np.diff(start) > 0
    matched = np.zeros(rest.height, bool); matched[b] = True
    # (country, name)-sorted mix of anchors (+) and distractors (-) for hard batches
    key = pl.concat([s1.select("country", "name").with_columns(pl.lit(1).alias("kind"), pl.int_range(pl.len()).alias("i")),
                     rest.select("country", "name").with_columns(pl.lit(0).alias("kind"), pl.int_range(pl.len()).alias("i"))
                     .filter(pl.Series(~matched))])
    key = key.filter((pl.col("kind") == 0) | pl.Series(np.concatenate([has, np.ones(int((~matched).sum()), bool)])))
    key = key.sort("country", "name")
    return dict(s1_txt=txt(s1), rest_txt=txt(rest), start=start, pos=b, anchors=np.where(has)[0],
                order_kind=key["kind"].to_numpy(), order_i=key["i"].to_numpy())


class Batches:
    def __init__(self, d, B, hard_frac, seed):
        self.d, self.B, self.hard, self.rng = d, B, hard_frac, np.random.default_rng(seed)
        self.perm, self.p = self.rng.permutation(d["anchors"]), 0

    def _pos(self, ai):
        st, en = self.d["start"][ai], self.d["start"][ai + 1]
        return self.d["pos"][st + (self.rng.random(len(ai)) * (en - st)).astype(np.int64)]

    def __next__(self):
        d, B = self.d, self.B
        ai = np.empty(0, np.int64)
        if self.rng.random() < self.hard:                         # contiguous window of the name-sorted mix
            n = len(d["order_i"])
            while len(ai) < B // 4:                                # windows of pure distractors are redrawn
                s = int(self.rng.integers(0, n - 2 * B))
                k, i = d["order_kind"][s:s + int(1.4 * B)], d["order_i"][s:s + int(1.4 * B)]
                ai, neg = i[k == 1][:B], i[k == 0][: B // 2]
        else:
            if self.p + B > len(self.perm): self.perm, self.p = self.rng.permutation(d["anchors"]), 0
            ai = self.perm[self.p:self.p + B]; self.p += B; neg = np.empty(0, np.int64)
        bi = self._pos(ai)
        return ([d["s1_txt"][i] for i in ai], [d["rest_txt"][i] for i in bi], [d["rest_txt"][i] for i in neg])


def embed(model, texts, prefix):
    f = model.preprocess([prefix + t for t in texts]); f = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in f.items()}
    return model(f)["sentence_embedding"]


def loss_fn(ea, ep, en, scale, dims):
    tot = 0.0
    for dm in dims:
        a = F.normalize(ea[:, :dm].float(), dim=-1); p = F.normalize(ep[:, :dm].float(), dim=-1)
        cand = torch.cat([p, F.normalize(en[:, :dm].float(), dim=-1)]) if len(en) else p
        lbl = torch.arange(len(a), device=a.device)
        tot = tot + F.cross_entropy(scale * a @ cand.T, lbl) + F.cross_entropy(scale * p @ a.T, lbl)
    return tot / (2 * len(dims))


def evaluate(model, prefix, dim=None):
    import json as _j
    T = _j.load(open(W + "texts.json")); texts = T["qt"] + T["pt"]
    q, pool, tp = (pl.read_parquet(W + f) for f in ("q.parquet", "pool.parquet", "tp.parquet"))
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        E = model.encode([prefix + x for x in texts], batch_size=512, convert_to_numpy=True, normalize_embeddings=False)
    model.train()
    out = {}
    for dm in ([dim] if dim else sorted({192, E.shape[1]})):
        e = E[:, :dm].astype(np.float32); e /= np.linalg.norm(e, axis=1, keepdims=True)
        out[f"d{dm}"] = recall(e, len(T["qt"]), q, pool, tp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--hard", type=float, default=0.5); ap.add_argument("--maxlen", type=int, default=64)
    ap.add_argument("--init", default=None, help="start from a saved fine-tuned dir"); ap.add_argument("--tag", default="")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--fit-parts", default=None, help="restrict anchors to these S1 partitions, e.g. E")
    a = ap.parse_args()
    hf, lic, prefix, kw = MODELS[a.model]
    out = W + f"ft_{a.model}{a.tag}"
    t0 = time.time(); d = load_data(a.fit_parts); print(f"data: {len(d['anchors'])} anchors, {len(d['pos'])} pairs, "
                                          f"{(d['order_kind'] == 0).sum()} distractors {time.time()-t0:.0f}s", flush=True)
    model = load_model(a.model, a.init); model.max_seq_length = a.maxlen
    full = model.get_sentence_embedding_dimension(); dims = sorted({192, full}) if full > 192 else [full]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    warm = min(500, a.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    scaler = torch.amp.GradScaler(); it = Batches(d, a.batch, a.hard, 0); model.train(); run = 0.0
    for step in range(1, a.steps + 1):
        qa, pp, nn_ = next(it)
        with torch.autocast("cuda", torch.float16):
            ea = embed(model, qa, prefix); ep = embed(model, pp + nn_, prefix)
            loss = loss_fn(ea, ep[:len(pp)], ep[len(pp):], a.scale, dims)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 100 == 0: print(f"step {step} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
        if a.eval_every and step % a.eval_every == 0 and step < a.steps:
            print("eval", step, json.dumps(evaluate(model, prefix)), flush=True)
    model.save(out)
    r = {"model": a.model, "tag": a.tag, "fit_parts": a.fit_parts, "steps": a.steps, "batch": a.batch, "lr": a.lr, "scale": a.scale, "hard": a.hard,
         "train_s": round(time.time() - t0), **evaluate(model, prefix)}
    print(json.dumps(r), flush=True)
    with open(W + "finetune_results.jsonl", "a") as f: f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
