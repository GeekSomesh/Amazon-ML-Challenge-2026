"""Stage-2 ensemble partners (same S1-entity folds as the stage-2 LightGBM):
an XGBoost (GPU) re-scorer and a second LightGBM with different capacity / seed, blended with stage 2.

  python stage2_ens.py train <stage1_tag> <s2_tag>     -> trains partners on the cached trainsim stage-2 frame,
                                                         prints validation F0.5 of every blend, writes oof_<s2>x / oof_<s2>b / oof_<s2>ens
  python stage2_ens.py test  <stage1_scored> <s2_tag> <out_tag>
                                                      -> scores test_scored_<stage1_scored> with the partners and blends
                                                         with test_scored_<s2_tag>: writes test_scored_<out_tag>
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb

from config import N_THREADS, WORK
from stage2 import MIN_P1
from train_stage2 import DROP, stage2_frame

XGB = dict(objective="binary:logistic", tree_method="hist", device="cuda", max_depth=8, eta=0.05,
           subsample=0.8, colsample_bytree=0.7, min_child_weight=10, reg_lambda=2.0, max_bin=256,
           eval_metric="logloss", seed=41)
LGB2 = dict(objective="binary", learning_rate=0.03, num_leaves=255, min_data_in_leaf=100,
            feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0,
            num_threads=N_THREADS, verbose=-1, seed=99)


def frame_path(split, tag):
    return os.path.join(WORK, f"s2frame_{split}_{tag}.parquet")


def train(stage1_tag, s2_tag):
    from decide import evaluate, one_owner, select_expected_f
    from eval_oof import load_gt_rows
    t = time.time()
    fp = frame_path("trainsim", stage1_tag)
    if os.path.exists(fp):
        f2 = pl.read_parquet(fp)
    else:
        oof = pl.read_parquet(os.path.join(WORK, f"oof_{stage1_tag}.parquet"))
        f2 = stage2_frame("trainsim", oof)
        f2.write_parquet(fp)
        del oof
    feats = [c for c in f2.columns if c not in DROP]
    print(f"frame {f2.height} pairs, {len(feats)} feats ({time.time() - t:.0f}s)", flush=True)
    px = np.zeros(f2.height, dtype=np.float32)
    pb = np.zeros(f2.height, dtype=np.float32)
    for fold in (0, 1):
        tr = f2.filter(pl.col("fold") != fold)
        is_val = ((tr["s1_row"].hash(seed=5) % 20) == 0).to_numpy()
        X = tr.select(feats).to_numpy().astype(np.float32)
        y = tr["label"].to_numpy().astype(np.float32)
        del tr
        msk = (f2["fold"] == fold).to_numpy()
        Xo = f2.filter(pl.col("fold") == fold).select(feats).to_numpy().astype(np.float32)
        # XGBoost partner
        dtr = xgb.QuantileDMatrix(X[~is_val], label=y[~is_val], max_bin=256)
        dva = xgb.QuantileDMatrix(X[is_val], label=y[is_val], ref=dtr)
        m = xgb.train(XGB, dtr, 4000, evals=[(dva, "valid")], early_stopping_rounds=80, verbose_eval=False)
        m.save_model(os.path.join(WORK, f"xgb_{s2_tag}x_fold{fold}.json"))
        px[msk] = m.predict(xgb.DMatrix(Xo), iteration_range=(0, m.best_iteration + 1))
        print(f"  xgb fold {fold}: best_iter={m.best_iteration} logloss={m.best_score:.5f} ({time.time() - t:.0f}s)", flush=True)
        del dtr, dva
        # second LightGBM
        dl = lgb.Dataset(X[~is_val], label=y[~is_val])
        dv = lgb.Dataset(X[is_val], label=y[is_val], reference=dl)
        b = lgb.train(LGB2, dl, 5000, valid_sets=[dv], callbacks=[lgb.early_stopping(80, verbose=False)])
        b.save_model(os.path.join(WORK, f"lgb_{s2_tag}b_fold{fold}.txt"))
        pb[msk] = b.predict(Xo, num_iteration=b.best_iteration)
        print(f"  lgb2 fold {fold}: iters={b.best_iteration} logloss={b.best_score['valid_0']['binary_logloss']:.5f} ({time.time() - t:.0f}s)", flush=True)
        del X, y, Xo, dl, dv
    base = pl.read_parquet(os.path.join(WORK, f"oof_{s2_tag}.parquet"), columns=["s1_row", "p_row", "label", "fold", "prob"])
    part = f2.select("s1_row", "p_row").with_columns(pl.Series("px", px), pl.Series("pb", pb))
    d = base.join(part, on=["s1_row", "p_row"], how="left").with_columns(
        pl.col("px").fill_null(pl.col("prob")), pl.col("pb").fill_null(pl.col("prob")))
    s1, gt = load_gt_rows("trainsim")
    rows = s1["s1_row"].cast(pl.UInt32).to_numpy()
    dd = d.filter(pl.col("prob") >= 0.002)

    def f05(expr):
        x = dd.with_columns(expr.alias("p"))
        return evaluate(select_expected_f(one_owner(x.select("s1_row", "p_row", pl.col("p").alias("prob")), "prob", "q"), "q"), gt, rows)
    res = {"s2 (lgb)": f05(pl.col("prob")), "xgb": f05(pl.col("px")), "lgb2": f05(pl.col("pb")),
           "s2+xgb": f05((pl.col("prob") + pl.col("px")) / 2), "s2+lgb2": f05((pl.col("prob") + pl.col("pb")) / 2),
           "s2+xgb+lgb2": f05((pl.col("prob") + pl.col("px") + pl.col("pb")) / 3)}
    for k, v in res.items():
        print(f"  [{k:12s}] owner+expF F0.5 = {v:.5f}", flush=True)
    d.select("s1_row", "p_row", "label", "fold", ((pl.col("prob") + pl.col("px") + pl.col("pb")) / 3).alias("prob")).write_parquet(
        os.path.join(WORK, f"oof_{s2_tag}ens.parquet"))
    d.select("s1_row", "p_row", "label", "fold", ((pl.col("prob") + pl.col("px")) / 2).alias("prob")).write_parquet(
        os.path.join(WORK, f"oof_{s2_tag}x2.parquet"))
    print(f"ENS_TRAIN_DONE ({time.time() - t:.0f}s)", flush=True)


def test(stage1_scored, s2_tag, out_tag, parts="xb"):
    t = time.time()
    # same columns / order as the frames the partners were trained on (oof_<stage1> carries blk_cos)
    scored = pl.read_parquet(os.path.join(WORK, f"test_scored_{stage1_scored}.parquet"), columns=["s1_row", "p_row", "blk_cos", "prob"])
    ft = stage2_frame("test", scored)
    feats = [c for c in ft.columns if c not in DROP]
    X = ft.select(feats).to_numpy().astype(np.float32)
    nf = lgb.Booster(model_file=os.path.join(WORK, f"lgb_{s2_tag.rstrip('f')}b_fold0.txt")).num_feature()
    assert X.shape[1] == nf, f"stage feature mismatch {X.shape[1]} vs {nf}"
    preds = []
    if "x" in parts:
        d = xgb.DMatrix(X)
        ms = []
        for f in (0, 1):
            m = xgb.Booster()
            m.load_model(os.path.join(WORK, f"xgb_{s2_tag.rstrip('f')}x_fold{f}.json"))
            ms.append(m)
        preds.append(np.mean([m.predict(d, iteration_range=(0, m.best_iteration + 1)) for m in ms], axis=0))
    if "b" in parts:
        ms = [lgb.Booster(model_file=os.path.join(WORK, f"lgb_{s2_tag.rstrip('f')}b_fold{f}.txt")) for f in (0, 1)]
        preds.append(np.mean([m.predict(X) for m in ms], axis=0))
    part = ft.select("s1_row", "p_row").with_columns(pl.Series("pe", np.mean(preds, axis=0).astype(np.float32)))
    base = pl.read_parquet(os.path.join(WORK, f"test_scored_{s2_tag}.parquet"), columns=["s1_row", "p_row", "prob"])
    k = 1 + len(preds)
    out = (base.join(part, on=["s1_row", "p_row"], how="left")
               .with_columns(((pl.col("prob") + (k - 1) * pl.col("pe").fill_null(pl.col("prob"))) / k).cast(pl.Float32).alias("prob"))
               .select("s1_row", "p_row", "prob"))
    out.write_parquet(os.path.join(WORK, f"test_scored_{out_tag}.parquet"))
    print(f"ENS_TEST_DONE test_scored_{out_tag} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], sys.argv[3])
    else:
        test(*sys.argv[2:6])
