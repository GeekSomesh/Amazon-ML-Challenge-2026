"""Adversarial validation: can a classifier tell TEST candidate pairs from TRAINSIM candidate pairs?
Restricted to the decision-relevant region (stage-1 ensemble prob >= 0.05) and to US / India (France has no
training counterpart). A high AUC means the simulation differs from the test set; the top features say how.

  python adversarial.py e12
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import glob
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from config import WORK, norm_path
from train_lgb import feature_names


def sample(split, scored, feats, n, seed):
    keep = scored.filter(pl.col("prob") >= 0.05).select("s1_row", "p_row")
    out = []
    for p in sorted(glob.glob(os.path.join(WORK, f"feat_{split}", "part_*.parquet"))):
        out.append(pl.read_parquet(p, columns=["s1_row", "p_row"] + feats).join(keep, on=["s1_row", "p_row"], how="semi"))
    d = pl.concat(out)
    return d.sample(min(n, d.height), seed=seed)


def main(tag="e12", n=600_000):
    feats = feature_names("trainsim")
    if os.environ.get("ADV_CONTEXT_ONLY"):
        feats = [f for f in feats if f.startswith("blk_")]
        print("context-only features:", feats, flush=True)
    if os.environ.get("ADV_CONTENT_ONLY"):
        feats = [f for f in feats if not (f.endswith("maxidf") or "_fp_" in f or "_f1_" in f or f.endswith("_fp") or f.endswith("common") or f.endswith("_miss")
                                           or f.startswith("blk_") or f in ("p_name_unseen",))]
        print("content-only features:", len(feats), flush=True)
    ctry = lambda split: pl.read_parquet(norm_path(split, 1), columns=["country"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
    te_sc = pl.read_parquet(os.path.join(WORK, f"test_scored_{tag}.parquet"), columns=["s1_row", "p_row", "prob"])
    only = os.environ.get("ADV_COUNTRY")
    te_sc = te_sc.join(ctry("test").filter((pl.col("country") == only) if only else (pl.col("country") != "France")), on="s1_row", how="semi")
    tr_sc = pl.scan_parquet(os.path.join(WORK, f"oof_{tag}.parquet")).select("s1_row", "p_row", "prob").filter(pl.col("prob") >= 0.05).collect(engine="streaming")
    if only:
        tr_sc = tr_sc.join(ctry("trainsim").filter(pl.col("country") == only), on="s1_row", how="semi")
    a = sample("trainsim", tr_sc, feats, n, 1).with_columns(pl.lit(0).alias("y"))
    b = sample("test", te_sc, feats, n, 2).with_columns(pl.lit(1).alias("y"))
    d = pl.concat([a, b]).sample(fraction=1.0, shuffle=True, seed=3)
    X = d.select(feats).to_numpy().astype(np.float32)
    y = d["y"].to_numpy()
    m = len(y) * 4 // 5
    dtr = lgb.Dataset(X[:m], label=y[:m])
    dva = lgb.Dataset(X[m:], label=y[m:], reference=dtr)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                  metric="auc", verbose=-1, num_threads=8, seed=5)
    b_ = lgb.train(params, dtr, 300, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])
    print(f"adversarial AUC (test vs trainsim, US+India, prob>=0.05): {b_.best_score['valid_0']['auc']:.4f}", flush=True)
    imp = sorted(zip(feats, b_.feature_importance("gain")), key=lambda x: -x[1])
    tot = sum(g for _, g in imp)
    print("top separating features:", [(f, round(100 * g / tot, 1)) for f, g in imp[:15]], flush=True)
    for f, _ in imp[:6]:
        qa = np.nanquantile(a[f].to_numpy().astype(float), [0.1, 0.25, 0.5, 0.75, 0.9])
        qb = np.nanquantile(b[f].to_numpy().astype(float), [0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"  {f:22s} trainsim q10-90 {np.round(qa, 3)}  test {np.round(qb, 3)}  mean {a[f].mean():.4f} vs {b[f].mean():.4f}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
