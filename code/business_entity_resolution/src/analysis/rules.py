"""Explicit generator rules on top of the model probabilities (decision layer).

Each rule encodes a pattern read off the data (see the documentation) and only moves a probability
when its condition holds; every rule is measured on the out-of-fold training predictions (per fold)
before it is used, and applied identically to every test record.

  R_heavy : the record carries heavy per-record noise (blank address / domain name / alias). In training
            96-99.99% of such records are a true copy of SOME Source-1 entity, and planted look-alikes never
            carry this noise -> if this S1 is the record's clear best owner, raise its probability.
  R_up    : planted look-alike branches sit a few doors UP (pool number = S1 number + 1..20) and repeat that
            number over their own records; true copies with number noise move DOWN / drop digits.
  R_down  : number one or two lower, or one digit dropped, with the same core name -> per-record noise.

  python rules.py <oof_tag>        -> F0.5 with / without each rule, per fold and overall
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import glob
import os
import sys

import numpy as np
import polars as pl

from config import WORK, norm_path
from decide import evaluate, one_owner, select_expected_f

FEAT = ["s1_row", "p_row", "n_tset", "a_tset", "p_name_unseen", "p_marker_extra", "num_first_dlev",
        "num_first_lendiff", "num_first_in"]


def record_info(split):
    """first house number per record (S1 then pool), heavy-noise flag and name_core for pool records"""
    def num(src):
        return (pl.read_parquet(norm_path(split, src), columns=["addr_full"])
                  .select(pl.col("addr_full").str.extract(r"\b(\d{1,9})\b", 1).cast(pl.Int64).fill_null(-1)).to_series().to_numpy())
    s1n = num(1)
    pool = pl.concat([pl.read_parquet(norm_path(split, s), columns=["addr_empty", "is_domain", "has_alias"]) for s in (2, 3)])
    heavy = (pool["addr_empty"] | pool["is_domain"] | pool["has_alias"]).to_numpy()
    pn = np.concatenate([num(2), num(3)])
    return s1n, pn, heavy


def attach(split, sc: pl.DataFrame, info=None) -> pl.DataFrame:
    s1n, pn, heavy = info or record_info(split)
    keys = sc.select("s1_row", "p_row")
    f = pl.concat([pl.read_parquet(p, columns=FEAT).join(keys, on=["s1_row", "p_row"], how="semi")
                   for p in sorted(glob.glob(os.path.join(WORK, f"feat_{split}", "part_*.parquet")))])
    d = sc.join(f, on=["s1_row", "p_row"], how="left")
    sr = d["s1_row"].to_numpy().astype(np.int64)
    pr = d["p_row"].to_numpy().astype(np.int64)
    return d.with_columns(pl.Series("_n1", s1n[sr]), pl.Series("_n2", pn[pr]), pl.Series("_heavy", heavy[pr]))


def apply_rules(d: pl.DataFrame, rules=("heavy", "up", "down"), col="prob") -> pl.DataFrame:
    p = pl.col(col)
    both = (pl.col("_n1") >= 0) & (pl.col("_n2") >= 0)
    sd = pl.col("_n2") - pl.col("_n1")
    d = d.with_columns(
        p.rank("ordinal", descending=True).over("p_row").alias("_rk_p"),
        pl.len().over("s1_row", "_n2").alias("_n_same_num"),
    )
    out = p
    if "up" in rules:
        # look-alike branch: a few doors up, name not identical, number repeated by another candidate of this S1
        cond = both & (sd >= 1) & (sd <= 20) & (pl.col("n_tset") < 100) & (pl.col("_n_same_num") >= 2) & (pl.col("num_first_in") == 0)
        out = pl.when(cond).then(out * 0.3).otherwise(out)
    if "down" in rules:
        cond = both & (((sd >= -2) & (sd <= -1)) | ((pl.col("num_first_dlev") == 1) & (pl.col("num_first_lendiff") == 1) & (sd < 0))) \
            & (pl.col("n_tset") >= 100) & (pl.col("a_tset") >= 90)
        out = pl.when(cond).then(1 - (1 - out) * 0.5).otherwise(out)
    if "heavy" in rules:
        cond = pl.col("_heavy") & (pl.col("_rk_p") == 1) & (pl.col("n_tset") >= 90) & (p >= 0.3)
        out = pl.when(cond).then(1 - (1 - out) * 0.5).otherwise(out)
    return d.with_columns(out.alias(col)).drop("_rk_p", "_n_same_num")


def main(tag):
    from eval_oof import load_gt_rows
    oof = (pl.scan_parquet(os.path.join(WORK, f"oof_{tag}.parquet")).select("s1_row", "p_row", "fold", "prob")
             .filter(pl.col("prob") >= 0.002).collect(engine="streaming"))
    s1, gt = load_gt_rows("trainsim")
    d = attach("trainsim", oof)
    folds = {f: s1.with_columns(pl.col("s1_row").cast(pl.UInt32)).join(oof.select("s1_row", "fold").unique("s1_row"), on="s1_row")
                 .filter(pl.col("fold") == f)["s1_row"].to_numpy() for f in (0, 1)}

    def score(x):
        sel = select_expected_f(one_owner(x.select("s1_row", "p_row", "prob"), "prob", "q"), "q")
        out = {}
        for f, rows in folds.items():
            rs = pl.Series(rows).implode()
            out[f] = evaluate(sel.filter(pl.col("s1_row").is_in(rs)), gt.filter(pl.col("s1_row").is_in(rs)), rows)
        out["all"] = evaluate(sel, gt, s1["s1_row"].cast(pl.UInt32).to_numpy())
        return out

    base = score(d)
    print(f"[{tag}] base: " + "  ".join(f"{k}={v:.5f}" for k, v in base.items()), flush=True)
    for r in (("heavy",), ("up",), ("down",), ("heavy", "up", "down")):
        s = score(apply_rules(d, r))
        print(f"   +{'+'.join(r):16s}: " + "  ".join(f"{k}={v:.5f} ({v - base[k]:+.5f})" for k, v in s.items()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
