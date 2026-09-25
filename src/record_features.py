"""Record-level caches and the pair features built from them (additive blocker columns, schema v3).

Unsupervised and per country (IDF and frequencies are computed on the split's own S1 + S2/S3 text, so they exist
for unseen countries such as France; no labels are used).

pair features
  name_wjacc / addr_wjacc     IDF-weighted token Jaccard of core names / addresses
  name_rare_miss_{s1,r}       max IDF of a core-name token present on one side only (a distinctive word missing)
  addr_rare_miss_{s1,r}       same for address tokens
  name_idf_match              summed IDF of shared core-name tokens (absolute evidence, not a ratio)
record features (repeated per edge)
  nfreq_s1 / nfreq_r          records in the country (S1 + S2/S3) sharing this core name: chains and generic names
  afreq_s1 / afreq_r          records sharing this exact normalised address: malls, office towers, registered agents
  r_nonlatin                  candidate's raw name or address contains non-Latin letters (transliterated upstream)"""
import numpy as np, polars as pl
from pair_features import core_name


def _tok(s: pl.Series) -> pl.Series:
    return s.str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() > 0)).list.unique()


class CountryRecords:
    """Per-country token IDF tables and frequency counts for S1 rows qi and R rows ri."""

    def __init__(self, s1, r, qi, ri):
        n1 = core_name(s1["name"][qi]); nr = core_name(r["name"][ri])
        a1 = s1["addr"][qi].str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()
        ar = r["addr"][ri].str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()
        self.nt1, self.ntr, self.at1, self.atr = _tok(n1), _tok(nr), _tok(a1), _tok(ar)
        self.n_idf = self._idf(pl.concat([self.nt1, self.ntr]))
        self.a_idf = self._idf(pl.concat([self.at1, self.atr]))
        allname = pl.concat([n1, nr]); alladdr = pl.concat([a1, ar])
        vc = lambda x: x.alias("k").value_counts(name="n")
        nc, ac = vc(allname), vc(alladdr)
        f = lambda s, cnt, blank: pl.DataFrame({"k": s}).join(cnt, on="k", how="left", maintain_order="left") \
            .select(pl.when(pl.col("k").str.len_chars() < blank).then(0).otherwise(pl.col("n"))).to_series().to_numpy().astype(np.int32)
        self.nfreq1, self.nfreqr = f(n1, nc, 2), f(nr, nc, 2)
        self.afreq1, self.afreqr = f(a1, ac, 8), f(ar, ac, 8)
        raw = (r["business_name"][ri] + " " + r["business_address"][ri])
        self.r_nonlatin = raw.str.contains(r"[^\p{Latin}\p{Common}\p{Inherited}]").fill_null(False).to_numpy()

    @staticmethod
    def _idf(toks):
        n = len(toks)
        df = toks.explode().drop_nulls().alias("t").value_counts(name="df")
        return df.with_columns((np.log((n + 1) / (pl.col("df") + 1)) + 1).cast(pl.Float32).alias("w")).select("t", "w")

    @staticmethod
    def _overlap(t1, tr, idf, pre):
        """IDF-weighted Jaccard, shared IDF mass, and max IDF of one-sided tokens, per aligned pair."""
        n = len(t1)
        d = pl.DataFrame({"row": np.arange(n, dtype=np.int32), "a": t1, "b": tr})
        parts = {"i": d.select("row", pl.col("a").list.set_intersection("b").alias("t")),
                 "u": d.select("row", pl.col("a").list.set_union("b").alias("t")),
                 "x1": d.select("row", pl.col("a").list.set_difference("b").alias("t")),
                 "xr": d.select("row", pl.col("b").list.set_difference("a").alias("t"))}
        out = {}
        for k, df in parts.items():
            e = df.explode("t").drop_nulls("t").join(idf, on="t", how="left").with_columns(pl.col("w").fill_null(1.0))
            agg = e.group_by("row").agg(pl.col("w").sum().alias("s"), pl.col("w").max().alias("m"))
            full = pl.DataFrame({"row": np.arange(n, dtype=np.int32)}).join(agg, on="row", how="left").fill_null(0.0).sort("row")
            out[k] = (full["s"].to_numpy(), full["m"].to_numpy())
        u = out["u"][0]
        return {f"{pre}_wjacc": np.where(u > 0, out["i"][0] / np.maximum(u, 1e-9), -1).astype(np.float32),
                f"{pre}_rare_miss_s1": out["x1"][1].astype(np.float32), f"{pre}_rare_miss_r": out["xr"][1].astype(np.float32),
                **({f"{pre}_idf_match": out["i"][0].astype(np.float32)} if pre == "name" else {})}

    def pair(self, i, j):
        """i, j: local S1 / R indices (aligned). Returns dict of feature arrays."""
        f = self._overlap(self.nt1[i], self.ntr[j], self.n_idf, "name")
        f.update(self._overlap(self.at1[i], self.atr[j], self.a_idf, "addr"))
        f.update(nfreq_s1=self.nfreq1[i], nfreq_r=self.nfreqr[j], afreq_s1=self.afreq1[i], afreq_r=self.afreqr[j],
                 r_nonlatin=self.r_nonlatin[j])
        return f
