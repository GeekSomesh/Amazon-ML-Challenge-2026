"""Final stage 2: the stage-2 re-scorer with multilingual (LaBSE) similarity features.

Features added to the stage-2 frame (computed by emb_pairs.py and emb_sib.py):
  emb_*      name embeddings: cosine, its rank / gap among the S1's candidates and among the pool record's owners
  emb_sib_*  name similarity to the S1's other confident candidates (overall / same source)
  emba_*     the same pair features for address embeddings

  python stage2_emb.py train [stage1_tag]            -> lgb_<TAG>_fold{0,1}.txt, oof_<TAG>.parquet, validation F0.5
  python stage2_emb.py test <stage1_tag> <out_tag>   -> test_scored_<out_tag>.parquet

The final model uses all three groups (TAG = s12a). EMB_SIB=0 / EMB_ADDR=0 switch the sibling / address groups off
(ablations, TAG s12m / s12n).
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from config import WORK
from stage2 import MIN_P1
from train_stage2 import DROP, PARAMS, cached_frame, stage2_frame

SIB = os.environ.get("EMB_SIB", "1") != "0"
ADDR = os.environ.get("EMB_ADDR", "1") != "0"
TAG = "s12a" if ADDR else ("s12n" if SIB else "s12m")
EMB = ["emb_cos", "emb_rank_s1", "emb_gap_s1", "emb_gap_p", "emb_sib_max", "emb_sib_same_src", "emb_sib_w",
       "emba_cos", "emba_rank_s1", "emba_gap_s1", "emba_gap_p"]


def add_embedding_features(frame: pl.DataFrame, split: str) -> pl.DataFrame:
    keys = ["s1_row", "p_row"]
    frame = frame.join(pl.read_parquet(os.path.join(WORK, f"emb_feat_{split}.parquet")), on=keys, how="left")
    if SIB:
        frame = frame.join(pl.read_parquet(os.path.join(WORK, f"emb_sib_{split}.parquet")), on=keys, how="left")
    if ADDR:
        frame = frame.join(pl.read_parquet(os.path.join(WORK, f"emba_feat_{split}.parquet")), on=keys, how="left")
    return frame


def train(stage1_tag="e12"):
    from decide import evaluate, one_owner, select_expected_f
    from eval_oof import load_gt_rows
    t = time.time()
    f2 = add_embedding_features(pl.read_parquet(cached_frame(stage1_tag)), "trainsim")
    feats = [c for c in f2.columns if c not in DROP]
    json.dump(feats, open(os.path.join(WORK, f"lgb_{TAG}_features.json"), "w"))
    print(f"frame {f2.height} pairs, {len(feats)} features ({time.time() - t:.0f}s)", flush=True)
    prob = np.zeros(f2.height, dtype=np.float32)
    models = []
    for fold in (0, 1):
        tr = f2.filter(pl.col("fold") != fold)
        is_val = (tr["s1_row"].hash(seed=5) % 20) == 0
        a, v = tr.filter(~is_val), tr.filter(is_val)
        dtr = lgb.Dataset(a.select(feats).to_numpy().astype(np.float32), label=a["label"].to_numpy())
        dva = lgb.Dataset(v.select(feats).to_numpy().astype(np.float32), label=v["label"].to_numpy(), reference=dtr)
        del a, v, tr
        m = lgb.train(PARAMS, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        m.save_model(os.path.join(WORK, f"lgb_{TAG}_fold{fold}.txt"))
        msk = (f2["fold"] == fold).to_numpy()
        prob[msk] = m.predict(f2.filter(pl.col("fold") == fold).select(feats).to_numpy().astype(np.float32), num_iteration=m.best_iteration)
        models.append(m)
        print(f"fold {fold}: iters={m.best_iteration} logloss={m.best_score['valid_0']['binary_logloss']:.5f} ({time.time() - t:.0f}s)", flush=True)
    imp = sorted(zip(feats, models[0].feature_importance("gain")), key=lambda x: -x[1])
    print("embedding feature importance ranks:", [(i + 1, f) for i, (f, _) in enumerate(imp) if f in EMB], flush=True)
    # out-of-fold probabilities: stage-2 pairs get the new model, pairs below MIN_P1 keep their stage-1 probability
    base = pl.read_parquet(os.path.join(WORK, f"oof_{stage1_tag}.parquet"), columns=["s1_row", "p_row", "label", "fold", "prob"])
    res = f2.select("s1_row", "p_row").with_columns(pl.Series("pm", prob))
    d = base.join(res, on=["s1_row", "p_row"], how="left").with_columns(pl.col("pm").fill_null(pl.col("prob")))
    d.select("s1_row", "p_row", "label", "fold", pl.col("pm").alias("prob")).write_parquet(os.path.join(WORK, f"oof_{TAG}.parquet"))
    # validation macro F0.5 (owner + expected-F decision), overall and per country
    s1, gt = load_gt_rows("trainsim")
    dd = d.filter(pl.col("pm") >= 0.002)
    c = s1.select(pl.col("s1_row").cast(pl.UInt32), "country")
    for ctry in [None] + sorted(c["country"].unique().to_list()):
        r = c if ctry is None else c.filter(pl.col("country") == ctry)
        rs = r["s1_row"].implode()
        sub = dd.filter(pl.col("s1_row").is_in(rs))
        sel = select_expected_f(one_owner(sub.select("s1_row", "p_row", pl.col("pm").alias("prob")), "prob", "q"), "q")
        f = evaluate(sel, gt.filter(pl.col("s1_row").is_in(rs)), r["s1_row"].to_numpy())
        print(f"  validation macro F0.5 [{ctry or 'all'}] = {f:.5f}", flush=True)
    print(f"done in {time.time() - t:.0f}s", flush=True)


def test(stage1_tag, out_tag):
    t = time.time()
    feats = json.load(open(os.path.join(WORK, f"lgb_{TAG}_features.json")))
    scored = pl.read_parquet(os.path.join(WORK, f"test_scored_{stage1_tag}.parquet"), columns=["s1_row", "p_row", "blk_cos", "prob"])
    ft = add_embedding_features(stage2_frame("test", scored), "test")
    X = ft.select(feats).to_numpy().astype(np.float32)
    models = [lgb.Booster(model_file=os.path.join(WORK, f"lgb_{TAG}_fold{f}.txt")) for f in (0, 1)]
    assert X.shape[1] == models[0].num_feature(), "feature mismatch between training and test frames"
    pt = np.mean([m.predict(X) for m in models], axis=0).astype(np.float32)
    res = ft.select("s1_row", "p_row").with_columns(pl.Series("prob", pt))
    low = scored.filter(pl.col("prob") < MIN_P1).select("s1_row", "p_row", pl.col("prob").cast(pl.Float32))
    pl.concat([res, low]).write_parquet(os.path.join(WORK, f"test_scored_{out_tag}.parquet"))
    print(f"wrote test_scored_{out_tag}.parquet ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(*sys.argv[2:3])
    else:
        test(*sys.argv[2:4])
