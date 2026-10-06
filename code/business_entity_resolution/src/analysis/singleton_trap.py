"""Singleton traps: an S1 whose selected pool records ALL carry a house number different from the S1's
(both numbers present) - is it a singleton lured by a planted look-alike group?

  python singleton_trap.py <oof_tag> <test_tag>
validation: share of such S1 that are true singletons + F0.5 of predicting empty for them instead;
test: how often such S1 occur per country (vs validation)."""
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
from eval_oof import load_gt_rows


def selection(scored):
    oo = one_owner(scored.select("s1_row", "p_row", "prob"), "prob", "q")
    return select_expected_f(oo, "q").join(oo.select("s1_row", "p_row", "q"), on=["s1_row", "p_row"])


def trap_flags(split, sel):
    F = ["s1_row", "p_row", "num_first_in", "num_n1", "num_n2", "p_addr_empty", "n_tset", "n_full"]
    f = pl.concat([pl.read_parquet(p, columns=F).join(sel.select("s1_row", "p_row"), on=["s1_row", "p_row"], how="semi")
                   for p in sorted(glob.glob(os.path.join(WORK, f"feat_{split}", "part_*.parquet")))])
    d = sel.join(f, on=["s1_row", "p_row"], how="left")
    both = (pl.col("num_n1") > 0) & (pl.col("num_n2") > 0)
    g = d.group_by("s1_row").agg(
        pl.len().alias("k"),
        (both & (pl.col("num_first_in") == 0)).sum().alias("k_mis"),
        (both & (pl.col("num_first_in") > 0)).sum().alias("k_hit"),
        pl.col("q").max().alias("qmax"),
        (pl.col("n_tset") >= 100).sum().alias("k_name_ok"))
    if os.environ.get("TRAP") == "name":
        return g.with_columns((pl.col("k_name_ok") == 0).alias("trap"))
    return g.with_columns(((pl.col("k_mis") >= 1) & (pl.col("k_hit") == 0)).alias("trap"))


def main(oof_tag, test_tag):
    oof = pl.scan_parquet(os.path.join(WORK, f"oof_{oof_tag}.parquet")).select("s1_row", "p_row", "prob").filter(pl.col("prob") >= 0.002).collect(engine="streaming")
    s1, gt = load_gt_rows("trainsim")
    sel = selection(oof)
    g = trap_flags("trainsim", sel)
    ngt = gt.group_by("s1_row").len().rename({"len": "ngt"})
    g = g.join(ngt, on="s1_row", how="left").with_columns(pl.col("ngt").fill_null(0))
    c = s1.select(pl.col("s1_row").cast(pl.UInt32), "country")
    g = g.join(c, on="s1_row")
    print("VALIDATION: S1 with selections, trap = all selected records miss the S1 house number")
    print(g.group_by("country", "trap").agg(pl.len(), (pl.col("ngt") == 0).mean().round(4).alias("singleton_rate"),
                                            pl.col("k").mean().round(2).alias("mean_sel")).sort("country", "trap"))
    rows = s1["s1_row"].cast(pl.UInt32).to_numpy()
    base = evaluate(sel, gt, rows)
    for qcut in (0.0, 0.9, 0.97, 0.99):
        drop = g.filter(pl.col("trap") & (pl.col("qmax") < qcut if qcut > 0 else pl.lit(True)))
        new = evaluate(sel.join(drop.select("s1_row"), on="s1_row", how="anti"), gt, rows)
        print(f"  empty the trap S1 (all-mismatch, qmax<{qcut or 'any'}): {drop.height} S1  F0.5 {base:.5f} -> {new:.5f} ({new - base:+.5f})", flush=True)
    te = pl.scan_parquet(os.path.join(WORK, f"test_scored_{test_tag}.parquet")).select("s1_row", "p_row", "prob").filter(pl.col("prob") >= 0.002).collect(engine="streaming")
    tg = trap_flags("test", selection(te))
    ct = pl.read_parquet(norm_path("test", 1), columns=["country"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
    n = ct.group_by("country").len().rename({"len": "S1"})
    vn = c.group_by("country").len().rename({"len": "S1"})
    print("trap S1 per 1000 S1 - validation:", g.filter(pl.col("trap")).group_by("country").len().join(vn, on="country")
          .with_columns((1000 * pl.col("len") / pl.col("S1")).round(2).alias("per_1000")).select("country", "per_1000").sort("country").rows())
    print("trap S1 per 1000 S1 - test:      ", tg.filter(pl.col("trap")).join(ct, on="s1_row").group_by("country").len().join(n, on="country")
          .with_columns((1000 * pl.col("len") / pl.col("S1")).round(2).alias("per_1000")).select("country", "per_1000").sort("country").rows())


if __name__ == "__main__":
    main(*sys.argv[1:3])
