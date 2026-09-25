"""Derive a density-stress split from a train-derived one.

Test has 5.5-5.8 S2/S3 records per S1 in every country; train has 4.67. Dropping a fraction f of S1s turns their
matches into orphan distractors (realistic ones: real businesses with no S1), raising R/S1 to the test level:
f ~ 0.19 gives (1.21 + 3.46 f) / (1 - f) ~ 2.3 distractors per S1, i.e. about 5.8 records per S1.

  python make_split.py minid --base mini --drop-s1 0.19 --emb xsm

Writes work/{name}_source{1,2,3}.parquet and work/emb_{emb}_{name}_source{1,2,3}.npy (row-aligned with the parquets).
S2/S3 files are hard links to the base split's files (no copy)."""
import argparse, os, zlib, numpy as np, polars as pl

from config import WORK as W


def _link(src, dst):
    if os.path.exists(dst): os.remove(dst)
    os.link(src, dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--base", default="mini"); ap.add_argument("--drop-s1", type=float, default=0.19)
    ap.add_argument("--emb", action="append", default=[], help="embedding tag(s) to subset alongside"); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    s1 = pl.read_parquet(W + f"{a.base}_source1.parquet")
    u = np.array([zlib.crc32(f"{a.seed}:{e}".encode()) for e in s1["entity_id"].to_list()]) / 2**32
    keep = u >= a.drop_s1
    s1.filter(pl.Series(keep)).write_parquet(W + f"{a.name}_source1.parquet")
    for s in ("source2", "source3"): _link(W + f"{a.base}_{s}.parquet", W + f"{a.name}_{s}.parquet")
    for t in a.emb:
        np.save(W + f"emb_{t}_{a.name}_source1.npy", np.load(W + f"emb_{t}_{a.base}_source1.npy")[keep])
        for s in ("source2", "source3"): _link(W + f"emb_{t}_{a.base}_{s}.npy", W + f"emb_{t}_{a.name}_{s}.npy")
    r = sum(pl.read_parquet(W + f"{a.name}_{s}.parquet", columns=["entity_id"]).height for s in ("source2", "source3"))
    print(f"{a.name}: S1 {keep.sum()} of {len(keep)} kept, R {r}, R/S1 {r / keep.sum():.2f}")
