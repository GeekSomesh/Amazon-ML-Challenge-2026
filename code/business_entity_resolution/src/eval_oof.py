"""Evaluate decision strategies on out-of-fold probabilities of the trainsim split.
The ground truth includes pairs that blocking missed, so the score is the true macro F0.5.
"""
import os
import sys

import numpy as np
import polars as pl

from config import WORK, gt_path, norm_path
from decide import evaluate, exclusive, one_owner, select_expected_f, select_threshold


def load_gt_rows(split="trainsim"):
    s1 = pl.read_parquet(norm_path(split, 1), columns=["entity_id", "country"]).with_row_index("s1_row")
    pool = pl.concat([pl.read_parquet(norm_path(split, 2), columns=["entity_id"]),
                      pl.read_parquet(norm_path(split, 3), columns=["entity_id"])]).with_row_index("p_row")
    gt = pl.read_parquet(gt_path(split))
    ex = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
            .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "p_id"}))
    ex = (ex.join(s1.select(pl.col("entity_id").alias("s1_id"), "s1_row"), on="s1_id")
            .join(pool.select(pl.col("entity_id").alias("p_id"), "p_row"), on="p_id"))
    return s1, ex.select(pl.col("s1_row").cast(pl.UInt32), pl.col("p_row").cast(pl.UInt32))


def report(oof: pl.DataFrame, col="prob", split="trainsim"):
    s1, gt = load_gt_rows(split)
    rows = s1["s1_row"].cast(pl.UInt32).to_numpy()
    print(f"S1 entities {len(rows)}, GT pairs {gt.height}, candidate positives {int(oof['label'].sum())} "
          f"(blocking recall {oof['label'].sum() / gt.height:.4f})")
    ex = exclusive(oof, col)
    res = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        res[f"excl+thr{t}"] = evaluate(select_threshold(ex, t, col), gt, rows)
    res["thr0.5(no excl)"] = evaluate(select_threshold(oof, 0.5, col), gt, rows)
    res["excl+expF"] = evaluate(select_expected_f(ex, col), gt, rows)
    oo = one_owner(oof, col, "q")
    for t in (0.4, 0.5, 0.6):
        res[f"owner+thr{t}"] = evaluate(select_threshold(oo, t, "q"), gt, rows)
    res["owner+expF"] = evaluate(select_expected_f(oo, "q"), gt, rows)
    for k, v in res.items():
        print(f"  {k:18s} F0.5 = {v:.5f}")
    best = max(res, key=res.get)
    # per-country breakdown for the best strategy
    base, c2 = (oo, "q") if best.startswith("owner") else (ex if "no excl" not in best else oof, col)
    sel = select_expected_f(base, c2) if best.endswith("expF") else select_threshold(base, float(best.split("thr")[1].split("(")[0]), c2)
    for ctry in s1["country"].unique().to_list():
        r = s1.filter(pl.col("country") == ctry)["s1_row"].cast(pl.UInt32).to_numpy()
        print(f"  [{best}] {ctry}: {evaluate(sel.filter(pl.col('s1_row').is_in(pl.Series(r).implode())), gt, r):.5f}")
    return res


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "m1"
    col = sys.argv[2] if len(sys.argv) > 2 else "prob"
    report(pl.read_parquet(os.path.join(WORK, f"oof_{tag}.parquet")), col)
