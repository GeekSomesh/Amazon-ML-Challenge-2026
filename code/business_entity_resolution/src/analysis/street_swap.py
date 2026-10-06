"""Hypothesis: France noise replaces the STREET (same name, same house number, same city, other street).
Counts candidate pairs with identical core name + identical first house number but a different address,
per country on test (with the final decision), and the true-match rate of the same pattern on training labels.

  python street_swap.py <oof_tag> <test_tag>
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import glob
import os
import sys

import polars as pl

from config import WORK, norm_path
from decide import one_owner, select_expected_f

F = ["s1_row", "p_row", "n_tset", "num_first_eq", "a_tset", "ac_tset"]


def frame(split, scored):
    oo = one_owner(scored.select("s1_row", "p_row", "prob"), "prob", "q")
    sel = select_expected_f(oo, "q").with_columns(pl.lit(1).alias("sel"))
    f = pl.concat([pl.read_parquet(p, columns=F).filter((pl.col("n_tset") >= 100) & (pl.col("num_first_eq") > 0))
                   for p in sorted(glob.glob(os.path.join(WORK, f"feat_{split}", "part_*.parquet")))])
    d = f.join(scored, on=["s1_row", "p_row"], how="left").join(sel, on=["s1_row", "p_row"], how="left").with_columns(
        pl.col("sel").fill_null(0), pl.col("prob").fill_null(0.0))
    c = pl.read_parquet(norm_path(split, 1), columns=["country"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
    d = d.join(c, on="s1_row")
    return d.with_columns(pl.col("a_tset").cut([50, 70, 90]).alias("addr_sim"))


def main(oof_tag, test_tag):
    tr = pl.scan_parquet(os.path.join(WORK, f"oof_{oof_tag}.parquet")).select("s1_row", "p_row", "prob", "label").collect(engine="streaming")
    d = frame("trainsim", tr)
    pl.Config.set_tbl_rows(40)
    print("TRAIN (labels): same core name + same house number, by address similarity (a_tset)")
    print(d.group_by("country", "addr_sim").agg(pl.len(), pl.col("label").mean().round(4).alias("true_rate"),
                                                pl.col("prob").mean().round(3).alias("mean_p"), pl.col("sel").mean().round(3).alias("sel_rate")).sort("country", "addr_sim"))
    te = pl.read_parquet(os.path.join(WORK, f"test_scored_{test_tag}.parquet"), columns=["s1_row", "p_row", "prob"])
    t = frame("test", te)
    n1 = pl.read_parquet(norm_path("test", 1), columns=["country"]).group_by("country").len().rename({"len": "S1"})
    print("TEST: same core name + same house number, by address similarity")
    print(t.group_by("country", "addr_sim").agg(pl.len(), pl.col("prob").mean().round(3).alias("mean_p"), pl.col("sel").mean().round(3).alias("sel_rate"))
           .join(n1, on="country").with_columns((pl.col("len") / pl.col("S1")).round(4).alias("per_S1")).sort("country", "addr_sim"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
