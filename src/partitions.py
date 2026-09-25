"""Fixed S1 partitions of the training split, so that no learned stage is fitted on the labels it is scored on.

  H  holdout   the 2.5% eval world (work/q.parquet); never used for fitting anything
  M  matcher   MATCHER_FRAC of the remaining S1s; labels only for the selector (never seen by encoder or aliases)
  E  encoder   the rest; encoder fine-tuning and region-alias fitting

Writes work/partitions.parquet (entity_id, part). Deterministic (hash of the id), so it can be rebuilt anywhere.
Every S2/S3 record inherits the part of the S1 it matches; unmatched records have no part."""
import os, sys, zlib, numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(__file__))

from config import WORK as W
MATCHER_FRAC = 0.40
SEED = 20260925


def build():
    s1 = pl.read_parquet(W + "train_source1.parquet", columns=["entity_id", "country"])
    hq = pl.read_parquet(W + "q.parquet", columns=["entity_id"])["entity_id"]
    u = pl.Series(np.array([zlib.crc32(f"{SEED}:{e}".encode()) for e in s1["entity_id"].to_list()]) / 2**32)   # version-stable
    p = s1.with_columns(pl.when(pl.col("entity_id").is_in(hq.implode())).then(pl.lit("H"))
                        .when(u < MATCHER_FRAC).then(pl.lit("M")).otherwise(pl.lit("E")).alias("part"))
    p.select("entity_id", "part").write_parquet(W + "partitions.parquet")
    print(p.group_by("country", "part").len().sort("country", "part").rows())
    return p


def load(parts=None):
    """S1 ids (pl.Series) of the given parts, e.g. load('E'); all rows as a DataFrame when parts is None."""
    if not os.path.exists(W + "partitions.parquet"): build()
    p = pl.read_parquet(W + "partitions.parquet")
    return p if parts is None else p.filter(pl.col("part").is_in(list(parts)))["entity_id"]


def pairs(parts):
    """train_pairs restricted to S1s of the given parts."""
    return pl.read_parquet(W + "train_pairs.parquet").filter(pl.col("s1").is_in(load(parts).implode()))


if __name__ == "__main__":
    build()
