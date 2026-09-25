"""Normalise every train/test record once (Indic transliteration, NFKD, ligatures) -> work/{split}_{src}.parquet
columns: entity_id, country, name, addr (normalised), plus raw name/address for the matcher."""
import sys, time, multiprocessing as mp, polars as pl
from norm import norm
from config import DATA as D
from config import WORK as W
def rd(p): return pl.read_csv(D + p, separator="\t", quote_char=None, infer_schema_length=0).fill_null("")
def _norm_chunk(xs): return [norm(x) for x in xs]
def pmap(P, xs, n=20000): return [y for ch in P.imap(_norm_chunk, [xs[i:i + n] for i in range(0, len(xs), n)]) for y in ch]
if __name__ == "__main__":
    t0 = time.time()
    with mp.Pool(10) as P:
        for split in ("train", "test"):
            for s in ("source1", "source2", "source3"):
                df = rd(f"{split}/{split}_{s}.tsv")
                df = df.with_columns(pl.Series("name", pmap(P, df["business_name"].to_list())),
                                     pl.Series("addr", pmap(P, df["business_address"].to_list())))
                df.write_parquet(W + f"{split}_{s}.parquet")
                print(split, s, df.height, f"{time.time()-t0:.0f}s", flush=True)
    gt = rd("train/train_ground_truth.tsv")
    ex = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids") \
           .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
    ex.write_parquet(W + "train_pairs.parquet"); print("pairs", ex.height)
