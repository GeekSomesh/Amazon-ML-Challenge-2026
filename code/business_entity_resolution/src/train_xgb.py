"""Second stage-1 matcher: XGBoost (Apache-2.0) on the GPU, same 2-fold S1-entity cross-fitting,
trained on a larger sample than the LightGBM model; used as an ensemble partner.

  python train_xgb.py x4 m4   -> oof_x4, test_scored_x4, and the averaged ensemble oof_e4 / test_scored_e4
"""
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

from config import WORK
from train_lgb import feature_names, parts

PARAMS = dict(objective="binary:logistic", tree_method="hist", device="cuda", max_depth=9, eta=0.08,
              subsample=0.8, colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0, max_bin=256,
              eval_metric="logloss", seed=23)


def load(split, fold_out, frac, seed, feats):
    """part-by-part into float32 numpy (no full-size polars copy -> lower peak memory)"""
    Xs, ys, ss = [], [], []
    for p in parts(split):
        df = (pl.read_parquet(p, columns=["s1_row", "fold", "label"] + feats)
                .filter((pl.col("fold") != fold_out) & ((pl.col("s1_row").hash(seed=seed) % 1000) < int(frac * 1000))))
        Xs.append(df.select(feats).to_numpy().astype(np.float32))
        ys.append(df["label"].to_numpy().astype(np.float32))
        ss.append(df["s1_row"].to_numpy())
        del df
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss)


def main(tag="x4", lgb_tag="m4", frac=float(os.environ.get("BER_XGB_FRAC", "0.3")), rounds=3000):
    feats = feature_names("trainsim")
    with open(os.path.join(WORK, f"xgb_{tag}_features.json"), "w") as f:
        json.dump(feats, f)
    models = {}
    for fold in (0, 1):
        t = time.time()
        X, y, srow = load("trainsim", fold, frac, 31 + fold, feats)
        is_val = ((pl.Series(srow).hash(seed=77) % 20) == 0).to_numpy()
        dtr = xgb.QuantileDMatrix(X[~is_val], label=y[~is_val], max_bin=256)
        dva = xgb.QuantileDMatrix(X[is_val], label=y[is_val], ref=dtr)
        del X
        m = xgb.train(PARAMS, dtr, rounds, evals=[(dva, "valid")], early_stopping_rounds=60, verbose_eval=False)
        m.save_model(os.path.join(WORK, f"xgb_{tag}_fold{fold}.json"))
        models[fold] = m
        print(f"xgb fold {fold}: {int((~is_val).sum())} pairs, best_iter={m.best_iteration} "
              f"logloss={m.best_score:.5f} in {time.time() - t:.0f}s", flush=True)
        del dtr, dva
    # OOF on trainsim + ensemble with the LightGBM OOF
    lgb_oof = pl.read_parquet(os.path.join(WORK, f"oof_{lgb_tag}.parquet"))
    out = []
    for p in parts("trainsim"):
        df = pl.read_parquet(p, columns=["s1_row", "p_row", "fold"] + feats)
        pr = np.zeros(df.height, dtype=np.float32)
        fold = df["fold"].to_numpy()
        X = df.select(feats).to_numpy().astype(np.float32)
        for f in (0, 1):
            msk = fold == f
            if msk.any():
                pr[msk] = models[f].predict(xgb.DMatrix(X[msk]), iteration_range=(0, models[f].best_iteration + 1))
        out.append(df.select("s1_row", "p_row").with_columns(pl.Series("prob_x", pr)))
    ox = pl.concat(out)
    e = lgb_oof.join(ox, on=["s1_row", "p_row"], how="left")
    e.select("s1_row", "p_row", "label", "fold", "blk_cos", pl.col("prob_x").alias("prob")).write_parquet(os.path.join(WORK, f"oof_{tag}.parquet"))
    e.with_columns(((pl.col("prob") + pl.col("prob_x")) / 2).alias("prob")).drop("prob_x").write_parquet(
        os.path.join(WORK, f"oof_e{tag[1:]}.parquet"))
    # test scoring (average of both fold models) + ensemble with the LightGBM test scores
    lgb_test = pl.read_parquet(os.path.join(WORK, f"test_scored_{lgb_tag}.parquet"))
    out = []
    for p in sorted(glob.glob(os.path.join(WORK, "feat_test", "part_*.parquet"))):
        df = pl.read_parquet(p, columns=["s1_row", "p_row"] + feats)
        d = xgb.DMatrix(df.select(feats).to_numpy().astype(np.float32))
        pr = np.mean([models[f].predict(d, iteration_range=(0, models[f].best_iteration + 1)) for f in (0, 1)], axis=0)
        out.append(df.select("s1_row", "p_row").with_columns(pl.Series("prob_x", pr.astype(np.float32))))
    tx = pl.concat(out)
    t = lgb_test.join(tx, on=["s1_row", "p_row"], how="left")
    t.select("s1_row", "p_row", "blk_cos", pl.col("prob_x").alias("prob")).write_parquet(os.path.join(WORK, f"test_scored_{tag}.parquet"))
    t.with_columns(((pl.col("prob") + pl.col("prob_x")) / 2).alias("prob")).drop("prob_x").write_parquet(
        os.path.join(WORK, f"test_scored_e{tag[1:]}.parquet"))
    print("xgb + ensemble files written", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
