"""Re-score the test split with already-trained models (no retraining), e.g. after the test inputs were
re-normalised and feat_test/ rebuilt:

  python rescore_test.py <lgb_tag> <xgb_tag> <stage2_tag> [<stage3_tag>] [--suffix f]

  stage 1: LightGBM + XGBoost fold models, averaged per model, then ensembled (as in train_xgb.py)
  stage 2 (and optional stage 3): saved LightGBM fold models on the stage-2 feature frame
  France (unseen country): house-number noise-rate recalibration of the final stage
Writes test_scored_<tag><suffix>.parquet for every stage and test_scored_<last><suffix>r.parquet.
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import glob
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb

from config import WORK
from predict_test import score_parts
from recalibrate import recalibrate
from stage2 import MIN_P1
from train_stage2 import DROP, stage2_frame


def xgb_scores(tag):
    feats = json.load(open(os.path.join(WORK, f"xgb_{tag}_features.json")))
    models = []
    for f in (0, 1):
        b = xgb.Booster()
        b.load_model(os.path.join(WORK, f"xgb_{tag}_fold{f}.json"))
        models.append(b)
    out = []
    for p in sorted(glob.glob(os.path.join(WORK, "feat_test", "part_*.parquet"))):
        df = pl.read_parquet(p, columns=["s1_row", "p_row"] + feats)
        d = xgb.DMatrix(df.select(feats).to_numpy().astype(np.float32))
        pr = np.mean([m.predict(d, iteration_range=(0, m.best_iteration + 1)) for m in models], axis=0)
        out.append(df.select("s1_row", "p_row").with_columns(pl.Series("prob_x", pr.astype(np.float32))))
    return pl.concat(out)


def stage_scores(tag, scored):
    """apply saved stage-2 style fold models (averaged) to the stage-2 frame built on `scored`"""
    models = [lgb.Booster(model_file=os.path.join(WORK, f"lgb_{tag}_fold{f}.txt")) for f in (0, 1)]
    ft = stage2_frame("test", scored)
    feats = [c for c in ft.columns if c not in DROP]
    assert feats == models[0].feature_name() or len(feats) == models[0].num_feature(), "stage feature mismatch"
    X = ft.select(feats).to_numpy().astype(np.float32)
    pt = np.mean([m.predict(X) for m in models], axis=0).astype(np.float32)
    res = ft.select("s1_row", "p_row").with_columns(pl.Series("prob", pt))
    low = scored.filter(pl.col("prob") < MIN_P1).select("s1_row", "p_row", pl.col("prob").cast(pl.Float32))
    return pl.concat([res, low])


def main(argv):
    suf = "f"
    if "--suffix" in argv:
        i = argv.index("--suffix")
        suf = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    argv = [a for a in argv if a != "--reuse-stage1"]
    m, x, s2 = argv[:3]
    s3 = argv[3] if len(argv) > 3 else None
    t = time.time()
    etag = "e" + x[1:]
    epath = os.path.join(WORK, f"test_scored_{etag}{suf}.parquet")
    if os.environ.get("RESCORE_REUSE") and os.path.exists(epath):
        e = pl.read_parquet(epath)
        print(f"reusing {epath}", flush=True)
    else:
        lt = score_parts("test", m)
        lt.write_parquet(os.path.join(WORK, f"test_scored_{m}{suf}.parquet"))
        print(f"lgb {m}: {lt.height} pairs ({time.time() - t:.0f}s)", flush=True)
        e = lt.join(xgb_scores(x), on=["s1_row", "p_row"], how="left")
        e = e.with_columns(((pl.col("prob") + pl.col("prob_x")) / 2).alias("prob")).drop("prob_x")
        e.write_parquet(epath)
        print(f"ensemble {etag}{suf} written ({time.time() - t:.0f}s)", flush=True)
    # the stage-2 model was trained on a frame that carries the stage-1 blocking cosine (blk_cos) in the
    # same column position as here; later stages were trained without it
    last, sc = etag, e.select("s1_row", "p_row", "blk_cos", "prob")
    for tag in [s2] + ([s3] if s3 else []):
        sc = stage_scores(tag, sc)
        sc.write_parquet(os.path.join(WORK, f"test_scored_{tag}{suf}.parquet"))
        print(f"stage {tag}{suf} written ({time.time() - t:.0f}s)", flush=True)
        last = tag
    oof = pl.read_parquet(os.path.join(WORK, f"oof_{last}.parquet"), columns=["s1_row", "p_row", "prob"])
    recalibrate(sc, oof).write_parquet(os.path.join(WORK, f"test_scored_{last}{suf}r.parquet"))
    print(f"RESCORE_DONE test_scored_{last}{suf}r.parquet ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
