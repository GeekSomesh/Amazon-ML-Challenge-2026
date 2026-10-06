"""Stage-1 matcher: LightGBM binary classifier on pairwise + context features.

2-fold cross-fitting by Source-1 entity on the 'trainsim' split:
  model_f is trained on S1 entities of the other fold, so every candidate pair gets an
  out-of-fold probability that the decision layer can be tuned on without leakage.
"""
import glob
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from config import N_THREADS, WORK

ID_COLS = {"s1_row", "p_row", "label", "fold"}
# prediction early stopping: stop adding trees once 2|raw score| > 20 (p < 5e-5 or p > 0.99995); identical
# for OOF and test scoring, so stage 2 sees the same probability distribution
PRED_KW = dict(pred_early_stop=True, pred_early_stop_freq=20, pred_early_stop_margin=20.0)
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=255, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=2.0,
              max_bin=255, num_threads=N_THREADS, verbose=-1, seed=7)


def parts(split):
    return sorted(glob.glob(os.path.join(WORK, f"feat_{split}", "part_*.parquet")))


# All feature columns are used. p_marker_extra is built from English training markers only: French
# "sister-company" words also occur on true matches (public-leaderboard comparison).
EXCLUDE = set()


def feature_names(split):
    cols = pl.read_parquet_schema(parts(split)[0])
    return [c for c in cols if c not in ID_COLS and c not in EXCLUDE]


def load_training(split, fold_out, frac, seed, feats):
    """Pairs of S1 entities NOT in fold_out, subsampled by S1 entity (only needed columns)."""
    lf = pl.scan_parquet(parts(split)).filter(pl.col("fold") != fold_out)
    lf = lf.filter((pl.col("s1_row").hash(seed=seed) % 1000) < int(frac * 1000))
    return lf.select(["s1_row", "label"] + feats).collect()


def train(split="trainsim", frac=0.25, rounds=1000, tag="m1"):
    feats = feature_names(split)
    print(f"{len(feats)} features", flush=True)
    with open(os.path.join(WORK, f"lgb_{tag}_features.json"), "w") as f:
        json.dump(feats, f)
    models = {}
    for fold in (0, 1):
        t = time.time()
        df = load_training(split, fold, frac, 11 + fold, feats)
        # small early-stopping holdout (different S1 entities)
        is_val = (df["s1_row"].hash(seed=99) % 20) == 0
        tr, va = df.filter(~is_val), df.filter(is_val)
        del df
        print(f"fold {fold}: train {tr.height} pairs ({tr['label'].mean():.4f} pos), valid {va.height}", flush=True)
        ytr = tr["label"].to_numpy()
        Xtr = tr.select(feats).to_numpy().astype(np.float32)
        del tr
        dtr = lgb.Dataset(Xtr, label=ytr, free_raw_data=True)
        dva = lgb.Dataset(va.select(feats).to_numpy().astype(np.float32), label=va["label"].to_numpy(), reference=dtr)
        dtr.construct()
        del Xtr
        m = lgb.train(PARAMS, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
        m.save_model(os.path.join(WORK, f"lgb_{tag}_fold{fold}.txt"))
        models[fold] = m
        print(f"fold {fold}: best_iter={m.best_iteration} logloss={m.best_score['valid_0']['binary_logloss']:.5f} "
              f"in {time.time() - t:.0f}s", flush=True)
    imp = sorted(zip(feats, models[0].feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(f, round(g / 1e3)) for f, g in imp[:30]], flush=True)
    return models, feats


def oof_predict(models, feats, split="trainsim", tag="m1"):
    out = []
    for p in parts(split):
        df = pl.read_parquet(p)
        pr = np.zeros(df.height, dtype=np.float32)
        fold = df["fold"].to_numpy()
        X = df.select(feats).to_numpy().astype(np.float32)
        for f in (0, 1):
            msk = fold == f
            if msk.any():
                pr[msk] = models[f].predict(X[msk], num_iteration=models[f].best_iteration, **PRED_KW)
        out.append(df.select("s1_row", "p_row", "label", "fold", "blk_cos").with_columns(pl.Series("prob", pr)))
    oof = pl.concat(out)
    oof.write_parquet(os.path.join(WORK, f"oof_{tag}.parquet"))
    return oof


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "m1"
    models, feats = train(tag=tag)
    oof = oof_predict(models, feats, tag=tag)
    print("OOF rows", oof.height, flush=True)
