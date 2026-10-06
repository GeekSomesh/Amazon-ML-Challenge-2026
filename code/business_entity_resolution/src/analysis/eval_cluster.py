"""Validate the one-base-per-source cluster filter on OOF predictions (trainsim):
  python eval_cluster.py <tag>   -> macro F0.5 of owner+expF with and without cluster_filter, per country"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import glob
import os
import sys

import numpy as np
import polars as pl

from cluster_filter import cluster_filter
from config import WORK, norm_path
from decide import evaluate, one_owner, select_expected_f
from eval_oof import load_gt_rows


def first_numbers(split):
    def fn(src):
        return (pl.read_parquet(norm_path(split, src), columns=["addr_full"])
                  .select(pl.col("addr_full").str.extract(r"\b(\d{1,9})\b", 1).cast(pl.Int64).fill_null(-1))
                  .to_series().to_numpy())
    s1 = fn(1)
    pool = np.concatenate([fn(2), fn(3)])
    n2 = pl.scan_parquet(norm_path(split, 2)).select(pl.len()).collect().item()
    src = pl.Series(np.where(np.arange(len(pool)) < n2, "S2", "S3"))
    return np.concatenate([s1, pool]), src, len(s1)


def main(tag):
    split = "trainsim"
    oof = (pl.scan_parquet(os.path.join(WORK, f"oof_{tag}.parquet")).select("s1_row", "p_row", "prob")
             .filter(pl.col("prob") >= 0.002).collect(engine="streaming"))
    s1, gt = load_gt_rows(split)
    rows = s1["s1_row"].cast(pl.UInt32).to_numpy()
    oo = one_owner(oof, "prob", "q")
    sel = select_expected_f(oo, "q").join(oo.select("s1_row", "p_row", "q"), on=["s1_row", "p_row"])
    base = evaluate(sel, gt, rows)
    num, src, n1 = first_numbers(split)
    f = pl.concat([pl.read_parquet(p, columns=["s1_row", "p_row", "num_first_in"]).join(sel.select("s1_row", "p_row"), on=["s1_row", "p_row"], how="semi")
                   for p in sorted(glob.glob(os.path.join(WORK, f"feat_{split}", "part_*.parquet")))])
    sel2 = sel.join(f, on=["s1_row", "p_row"], how="left")
    s1_in_p = (sel2["num_first_in"].fill_null(0) > 0).cast(pl.Int8).to_numpy()
    filt = cluster_filter(sel2.drop("num_first_in"), num, src, n1, col="q", s1_in_p=s1_in_p)
    new = evaluate(filt, gt, rows)
    print(f"[{tag}] owner+expF F0.5 = {base:.5f}   with cluster filter = {new:.5f}   (delta {new - base:+.5f})", flush=True)
    c = s1.select("s1_row", "country").with_columns(pl.col("s1_row").cast(pl.UInt32))
    for ctry in sorted(c["country"].unique().to_list()):
        r = c.filter(pl.col("country") == ctry)["s1_row"].to_numpy()
        rs = pl.Series(r).implode()
        keep = lambda d: d.filter(pl.col("s1_row").is_in(rs))
        print(f"   {ctry}: {evaluate(keep(sel), keep(gt), r):.5f} -> {evaluate(keep(filt), keep(gt), r):.5f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
