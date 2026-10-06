"""Conservative variants on validation (idea #5): does requiring agreement between stage-2 models cost F0.5?
  python agreement.py        (needs oof_s12, oof_s12x2 = (s12+xgb)/2, oof_s12ens = (s12+xgb+lgb2)/3)"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import os

import polars as pl

from config import WORK
from decide import evaluate, one_owner, select_expected_f
from eval_oof import load_gt_rows


def load(tag):
    return (pl.scan_parquet(os.path.join(WORK, f"oof_{tag}.parquet")).select("s1_row", "p_row", pl.col("prob").alias(tag))
              .filter(pl.col(tag) >= 0.002).collect(engine="streaming"))


def sel(d, col):
    return select_expected_f(one_owner(d.select("s1_row", "p_row", pl.col(col).alias("prob")), "prob", "q"), "q")


def main():
    s1, gt = load_gt_rows("trainsim")
    rows = s1["s1_row"].cast(pl.UInt32).to_numpy()
    d = load("s12").join(load("s12x2"), on=["s1_row", "p_row"], how="full", coalesce=True).join(load("s12ens"), on=["s1_row", "p_row"], how="full", coalesce=True)
    d = d.with_columns([pl.col(c).fill_null(0.0) for c in ("s12", "s12x2", "s12ens")])
    # recover the xgb-only probability from the 2-way blend
    d = d.with_columns((2 * pl.col("s12x2") - pl.col("s12")).clip(0, 1).alias("xgb"))
    a, e = sel(d, "s12"), sel(d, "s12ens")
    res = {"s12": evaluate(a, gt, rows), "ens": evaluate(e, gt, rows),
           "min(s12,xgb)": evaluate(sel(d.with_columns(pl.min_horizontal("s12", "xgb").alias("m")), "m"), gt, rows),
           "sel(s12) AND sel(ens)": evaluate(a.join(e, on=["s1_row", "p_row"], how="semi"), gt, rows),
           "sel(s12) AND sel(xgb)": evaluate(a.join(sel(d, "xgb"), on=["s1_row", "p_row"], how="semi"), gt, rows)}
    for k, v in res.items():
        print(f"  {k:24s} F0.5 = {v:.5f} ({v - res['s12']:+.5f})", flush=True)


if __name__ == "__main__":
    main()
