"""Bulk-encode every record of a split with the (fine-tuned) encoder on the GPU in fp16.
Writes work/emb_{tag}_{split}_{src}.npy (float16, first --dim Matryoshka dims, L2-normalised), row-aligned with
work/{split}_{src}.parquet."""
import argparse, os, sys, time, numpy as np, polars as pl, torch
from sentence_transformers import SentenceTransformer
sys.path.insert(0, os.path.dirname(__file__))
from config import WORK as W


def encode_texts(model, texts, dim, prefix="", chunk=200_000, batch=1024):
    out = np.empty((len(texts), dim), np.float16)
    for s in range(0, len(texts), chunk):
        with torch.no_grad():
            e = model.encode([prefix + t for t in texts[s:s + chunk]], batch_size=batch, convert_to_tensor=True)
        e = torch.nn.functional.normalize(e[:, :dim].float(), dim=1)
        out[s:s + chunk] = e.half().cpu().numpy()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="train"); ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--prefix", default=""); ap.add_argument("--maxlen", type=int, default=64)
    a = ap.parse_args()
    m = SentenceTransformer(a.model, device="cuda", model_kwargs={"torch_dtype": torch.float16}, trust_remote_code=True)
    m.max_seq_length = a.maxlen
    for src in ("source1", "source2", "source3"):
        t = time.time(); df = pl.read_parquet(W + f"{a.split}_{src}.parquet", columns=["name", "addr"])
        E = encode_texts(m, (df["name"] + ", " + df["addr"]).to_list(), a.dim, a.prefix)
        np.save(W + f"emb_{a.tag}_{a.split}_{src}.npy", E)
        print(src, E.shape, f"{time.time()-t:.0f}s {len(E)/(time.time()-t):.0f} rec/s", flush=True)
