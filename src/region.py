"""Data-driven region membership for partitioned blocking — no hard-coded country or state lists.

Addresses are noisy and their components are shuffled (India especially), so a record gets a *set* of regions:
every address component (comma part, or the first/last one or two tokens of a part) that is in the country's
region vocabulary, directly or through a learned alias. Two records are compared by a partitioned leg if they
share a region, or if either has none (unknown -> searched against the whole country).

vocabulary (per country) = normalised comma parts seen in >= MIN_SHARE of that country's S1 records (S1 is the clean
reference source, so this works unsupervised on unseen countries such as France); aliases (train pairs of partition E only, see partitions.py, so holdout / matcher labels never leak in) map
frequent non-vocabulary parts of S2/S3 to a vocabulary term with >= 90% purity (e.g. 'up' -> 'uttar pradesh',
transliterated Indic state names, 'texas' -> 'tx')."""
import json, os, sys, numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(__file__))

from config import WORK as W
MIN_SHARE = 0.0005


def _parts_expr(col):
    """Normalised comma parts of a raw address (list[str])."""
    return (pl.col(col).str.to_lowercase().str.split(",")
            .list.eval(pl.element().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(r"\s+", " ").str.strip_chars())
            .list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)))


def _keys(parts):
    """Candidate region strings from parts: the part itself, its first/last token, its first/last two tokens."""
    out = set()
    for p in parts:
        t = p.split()
        out.update((p, t[0], t[-1], " ".join(t[:2]), " ".join(t[-2:])))
    return out


def build_vocab(s1):
    voc = {}
    for c in s1["country"].unique().to_list():
        x = s1.filter(pl.col("country") == c).select(_parts_expr("business_address").list.unique().alias("p")).explode("p")
        vc = x.group_by("p").len().filter((pl.col("len") >= MIN_SHARE * (s1["country"] == c).sum()) &
                                          ~pl.col("p").str.contains(r"\d"))
        voc[c] = sorted(vc["p"].to_list())
        print(f"region vocab {c}: {len(voc[c])} terms", flush=True)
    return voc


def learn_alias(s1, r, pairs, voc, full_pairs=7_638_365):
    min_n = max(12, round(30 * pairs.height / full_pairs))   # support threshold scaled to the share of labels used
    p1 = s1.select(pl.col("entity_id").alias("s1"), "country", _parts_expr("business_address").alias("P1"))
    pr = r.select(pl.col("entity_id").alias("m"), _parts_expr("business_address").alias("PR"))
    a = pairs.join(p1, on="s1").join(pr, on="m")
    alias = {}
    for c, v in voc.items():
        vs = pl.Series(v)
        x = a.filter(pl.col("country") == c).select(pl.col("P1").list.set_intersection(vs.implode()).alias("g"),
                                                    pl.col("PR").list.set_difference(vs.implode()).alias("o"))
        x = x.filter(pl.col("g").list.len() == 1).with_columns(pl.col("g").list.first()).explode("o").drop_nulls()
        g = x.group_by("o", "g").len().with_columns(pl.col("len").sum().over("o").alias("tot"))
        # purity of o -> g among pairs whose S1 has exactly one vocabulary region
        g = g.filter((pl.col("len") >= min_n) & (pl.col("len") >= 0.9 * pl.col("tot")) & ~pl.col("o").str.contains(r"\d"))
        alias[c] = dict(zip(g["o"].to_list(), g["g"].to_list()))
        print(f"region aliases {c}: {len(alias[c])}", flush=True)
    return alias


def assign(df, voc, alias):
    """-> list of region-sets (python sets of str), row-aligned with df."""
    parts = df.select(_parts_expr("business_address").alias("p"))["p"].to_list(); ctry = df["country"].to_list()
    vs = {c: set(v) for c, v in voc.items()}; out = []
    for c, ps in zip(ctry, parts):
        v = vs.get(c); al = alias.get(c, {})
        if not v: out.append(frozenset()); continue
        ks = _keys(ps)
        out.append(frozenset({k for k in ks if k in v} | {al[k] for k in ks if k in al}))
    return out


ALIAS_PARTS = "E"                                          # label partitions the aliases may be fitted on


def regions_for_split(split, s1, r):
    f = W + f"regions_{split}_{ALIAS_PARTS}.json"
    if os.path.exists(f):
        z = json.load(open(f)); memo = {}                  # intern: few distinct region sets, millions of records
        g = lambda xs: [memo.setdefault(tuple(x), frozenset(x)) for x in xs]
        g1 = g(z.pop("g1")); gr = g(z.pop("gr")); del z; return g1, gr
    vf = W + f"region_model_{ALIAS_PARTS}.json"
    if not os.path.exists(vf):
        from partitions import pairs as part_pairs
        tr1 = pl.read_parquet(W + "train_source1.parquet", columns=["entity_id", "country", "business_address"])
        trr = pl.concat([pl.read_parquet(W + f"train_{s}.parquet", columns=["entity_id", "business_address"]) for s in ("source2", "source3")])
        voc = build_vocab(tr1); alias = learn_alias(tr1, trr, part_pairs(ALIAS_PARTS), voc)   # vocab is unsupervised
        json.dump({"voc": voc, "alias": alias}, open(vf, "w"))
    m = json.load(open(vf)); voc, alias = m["voc"], m["alias"]
    if split != "train":                                   # countries unseen in training: vocabulary from this split's S1
        new = s1.filter(~pl.col("country").is_in(list(voc)))
        if new.height: voc.update(build_vocab(new))
    g1, gr = assign(s1, voc, alias), assign(r, voc, alias)
    json.dump({"g1": [sorted(x) for x in g1], "gr": [sorted(x) for x in gr]}, open(f, "w"))
    return g1, gr


def partition_cost(g1, gr, m1, mr):
    """Comparisons of the partitioned S1->R leg relative to a full-country search."""
    from collections import Counter
    c1 = Counter(g for x, m in zip(g1, m1) if m for g in x); cr = Counter(g for x, m in zip(gr, mr) if m for g in x)
    u1 = sum(1 for x, m in zip(g1, m1) if m and not x); ur = sum(1 for x, m in zip(gr, mr) if m and not x)
    n1, nr = int(np.sum(m1)), int(np.sum(mr))
    return (sum(c1[g] * cr.get(g, 0) for g in c1) + (n1 - u1) * ur + u1 * nr) / (n1 * nr)


if __name__ == "__main__":
    import time
    t0 = time.time(); split = sys.argv[1] if len(sys.argv) > 1 else "train"
    s1 = pl.read_parquet(W + f"{split}_source1.parquet", columns=["entity_id", "country", "business_address"])
    r = pl.concat([pl.read_parquet(W + f"{split}_{s}.parquet", columns=["entity_id", "country", "business_address"]) for s in ("source2", "source3")])
    g1, gr = regions_for_split(split, s1, r); print("assigned", f"{time.time()-t0:.0f}s", flush=True)
    for c in s1["country"].unique().to_list():
        m1 = (s1["country"] == c).to_numpy(); mr = (r["country"] == c).to_numpy()
        print(c, "S1 unknown", round(float(np.mean([not x for x, m in zip(g1, m1) if m])), 4),
              "R unknown", round(float(np.mean([not x for x, m in zip(gr, mr) if m])), 4),
              "mean regions/record", round(float(np.mean([len(x) for x, m in zip(gr, mr) if m])), 2),
              "S1->R cost vs full", round(partition_cost(g1, gr, m1, mr), 3))
    if split == "train":
        p = pl.read_parquet(W + "train_pairs.parquet")
        i1 = dict(zip(s1["entity_id"].to_list(), range(s1.height))); ir = dict(zip(r["entity_id"].to_list(), range(r.height)))
        ok = np.array([(not g1[i1[a]]) or (not gr[ir[b]]) or bool(g1[i1[a]] & gr[ir[b]]) for a, b in zip(p["s1"].to_list(), p["m"].to_list())])
        cc = s1.select("entity_id", "country").join(p.rename({"s1": "entity_id"}), on="entity_id")["country"]
        print("true pairs kept by partitioning:", round(float(ok.mean()), 4),
              {c: round(float(ok[(cc == c).to_numpy()].mean()), 4) for c in ("US", "India")})
