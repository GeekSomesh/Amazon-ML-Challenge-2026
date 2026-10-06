"""Train the stage-2 re-scorer on stage-1 out-of-fold probabilities (same S1-entity folds)
and score the test split.

  trainsim: oof_{tag1}.parquet (s1_row, p_row, label, fold, prob) -> stage-2 features -> OOF prob2
  test:     test_scored_{tag1}.parquet -> stage-2 features -> averaged fold models -> prob2
"""
import glob
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from config import N_THREADS, WORK
from features import Store
from stage2 import MIN_P1, build_stage2

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
              num_threads=N_THREADS, verbose=-1, seed=17)
DROP = {"s1_row", "p_row", "label", "fold", "prob"}


CARRY = ["name_f1_p", "name_f1_s1", "name_fp_s1", "addr_f1_s1", "p_addr_empty", "num_first_eq", "num_f1", "num_f2",
         "num_first_dlev", "a_tset", "ac_tset", "n_tset", "n_w1", "a_w1", "blk_margin_p", "p_is_domain",
         "num_absdiff", "num_lendiff", "num_first_absdiff", "num_first_lendiff", "name_fp_p", "addr_fp_p",
         "n_acronym", "n_join_full", "p_name_unseen", "p_marker_extra"]


def carry_stage1(split, f2):
    """Join a few stage-1 pairwise features onto the stage-2 frame (part by part)."""
    keys = f2.select("s1_row", "p_row")
    out = []
    for p in sorted(glob.glob(os.path.join(WORK, f"feat_{split}", "part_*.parquet"))):
        out.append(pl.read_parquet(p, columns=["s1_row", "p_row"] + CARRY).join(keys, on=["s1_row", "p_row"], how="semi"))
    return f2.join(pl.concat(out), on=["s1_row", "p_row"], how="left")


def stage2_frame(split, scored):
    st = Store(split)
    c = scored.rename({"prob": "prob1"})
    f2 = build_stage2(st, c)
    del st
    return carry_stage1(split, f2)


def cached_frame(stage1_tag="e12"):
    """Path of the trainsim stage-2 frame built on the stage-1 out-of-fold probabilities oof_<stage1_tag>
    (built once and cached in WORK, since it is shared by the embedding steps and the final stage 2)."""
    path = os.path.join(WORK, f"s2frame_trainsim_{stage1_tag}.parquet")
    if not os.path.exists(path):
        oof = pl.read_parquet(os.path.join(WORK, f"oof_{stage1_tag}.parquet"))
        stage2_frame("trainsim", oof).write_parquet(path)
    return path


def main(tag1="m1", tag2="s2"):
    t = time.time()
    oof = pl.read_parquet(os.path.join(WORK, f"oof_{tag1}.parquet"))
    f2 = stage2_frame("trainsim", oof)
    feats = [c for c in f2.columns if c not in DROP]
    print(f"stage-2 frame {f2.height} pairs (of {oof.height}), {len(feats)} features, built in {time.time() - t:.0f}s", flush=True)
    prob2 = np.zeros(f2.height, dtype=np.float32)
    models = []
    for fold in (0, 1):
        tr = f2.filter(pl.col("fold") != fold)
        is_val = (tr["s1_row"].hash(seed=5) % 20) == 0
        a, v = tr.filter(~is_val), tr.filter(is_val)
        dtr = lgb.Dataset(a.select(feats).to_numpy().astype(np.float32), label=a["label"].to_numpy())
        dva = lgb.Dataset(v.select(feats).to_numpy().astype(np.float32), label=v["label"].to_numpy(), reference=dtr)
        m = lgb.train(PARAMS, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        m.save_model(os.path.join(WORK, f"lgb_{tag2}_fold{fold}.txt"))
        msk = (f2["fold"] == fold).to_numpy()
        prob2[msk] = m.predict(f2.filter(pl.col("fold") == fold).select(feats).to_numpy().astype(np.float32),
                               num_iteration=m.best_iteration)
        models.append(m)
        print(f"fold {fold}: iters={m.best_iteration} logloss={m.best_score['valid_0']['binary_logloss']:.5f}", flush=True)
    imp = sorted(zip(feats, models[0].feature_importance("gain")), key=lambda x: -x[1])
    print("stage-2 top features:", [(f, round(g / 1e3)) for f, g in imp[:20]], flush=True)
    res = f2.select("s1_row", "p_row", "label", "fold").with_columns(pl.Series("prob", prob2))
    # pairs below MIN_P1 keep their (tiny) stage-1 probability
    low = oof.filter(pl.col("prob") < MIN_P1).select("s1_row", "p_row", "label", "fold", "prob")
    res = pl.concat([res, low.with_columns(pl.col("prob").cast(pl.Float32))])
    res.write_parquet(os.path.join(WORK, f"oof_{tag2}.parquet"))

    # ---- test
    scored = pl.read_parquet(os.path.join(WORK, f"test_scored_{tag1}.parquet"))
    ft = stage2_frame("test", scored)
    X = ft.select(feats).to_numpy().astype(np.float32)
    pt = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0).astype(np.float32)
    tres = ft.select("s1_row", "p_row").with_columns(pl.Series("prob", pt))
    tlow = scored.filter(pl.col("prob") < MIN_P1).select("s1_row", "p_row", "prob")
    pl.concat([tres, tlow.with_columns(pl.col("prob").cast(pl.Float32))]).write_parquet(os.path.join(WORK, f"test_scored_{tag2}.parquet"))
    print(f"stage-2 done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
