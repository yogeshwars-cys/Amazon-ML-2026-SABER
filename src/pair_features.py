"""String / structure features for every candidate pair (consumed by the matcher).
Vectorised with rapidfuzz.process.cpdist (multithreaded, pairwise over aligned lists)."""
import re, numpy as np, polars as pl
from rapidfuzz import process, fuzz, distance

LEGAL = (r"\b(private|pvt|limited|ltd|llc|l l c|inc|incorporated|corp|corporation|co|company|llp|plc|the|and|pc|pllc|lp|"
         r"sarl|sas|sa|eurl|sci|gmbh)\b")


def core_name(s: pl.Series) -> pl.Series:
    return s.str.replace_all(LEGAL, " ").str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()


def numbers(s: pl.Series) -> pl.Series:
    return s.str.extract_all(r"\d+").list.unique()


def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def string_features(n1, a1, nr, ar, g1=None, gr=None):
    """n1/a1: S1 normalised name/address (pl.Series), nr/ar: candidate side, aligned per pair."""
    c1, cr = core_name(n1), core_name(nr)
    f = {
        "name_ratio": _cp(n1.to_list(), nr.to_list(), fuzz.ratio),
        "name_tset": _cp(n1.to_list(), nr.to_list(), fuzz.token_set_ratio),
        "name_partial": _cp(c1.to_list(), cr.to_list(), fuzz.partial_ratio),
        "core_jw": _cp(c1.to_list(), cr.to_list(), distance.JaroWinkler.normalized_similarity),
        "core_equal": (c1 == cr).to_numpy() & (c1.str.len_chars() > 0).to_numpy(),
        "addr_tset": _cp(a1.to_list(), ar.to_list(), fuzz.token_set_ratio),
        "addr_tsort": _cp(a1.to_list(), ar.to_list(), fuzz.token_sort_ratio),
        "addr_partial": _cp(a1.to_list(), ar.to_list(), fuzz.partial_ratio),
        "name_len_diff": np.abs(n1.str.len_chars().to_numpy() - nr.str.len_chars().to_numpy()).astype(np.int16),
    }
    d1, dr = numbers(a1), numbers(ar)
    df = pl.DataFrame({"d1": d1, "dr": dr}).with_columns(
        pl.col("d1").list.set_intersection("dr").list.len().alias("i"),
        pl.col("d1").list.set_union("dr").list.len().alias("u"),
        pl.col("d1").list.first().alias("h1"), pl.col("dr").list.first().alias("hr"))
    f["num_jacc"] = np.where(df["u"].to_numpy() > 0, df["i"].to_numpy() / np.maximum(df["u"].to_numpy(), 1), -1).astype(np.float32)
    f["house_eq"] = np.where(df["h1"].is_null().to_numpy() | df["hr"].is_null().to_numpy(), -1,
                             (df["h1"] == df["hr"]).fill_null(False).to_numpy().astype(np.int8)).astype(np.int8)
    f["r_addr_empty"] = (ar.str.len_chars() == 0).to_numpy()
    if g1 is not None:
        f["region_overlap"] = np.array([(-1 if not a or not b else int(bool(a & b))) for a, b in zip(g1, gr)], np.int8)
    return f
