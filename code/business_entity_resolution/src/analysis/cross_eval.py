"""Cross-evaluate stage-1 model stacks on the CURRENT trainsim features (same S1-entity folds, so every
pair is scored by the fold model that did not see its S1 entity).

  python cross_eval.py m9:x9 m12:x12     -> macro F0.5 (one-owner + expected-F) of each LGB+XGB ensemble
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb

from config import WORK
from decide import evaluate, one_owner, select_expected_f
from eval_oof import load_gt_rows
from train_lgb import PRED_KW, parts


def score(lgb_tag, xgb_tag):
    lf = json.load(open(os.path.join(WORK, f"lgb_{lgb_tag}_features.json")))
    lm = [lgb.Booster(model_file=os.path.join(WORK, f"lgb_{lgb_tag}_fold{f}.txt")) for f in (0, 1)]
    xm, xf = None, None
    if xgb_tag:
        xf = json.load(open(os.path.join(WORK, f"xgb_{xgb_tag}_features.json")))
        xm = []
        for f in (0, 1):
            b = xgb.Booster()
            b.load_model(os.path.join(WORK, f"xgb_{xgb_tag}_fold{f}.json"))
            xm.append(b)
    out = []
    for p in parts("trainsim"):
        cols = sorted(set(["s1_row", "p_row", "fold"] + lf + (xf or [])))
        df = pl.read_parquet(p, columns=cols)
        fold = df["fold"].to_numpy()
        X = df.select(lf).to_numpy().astype(np.float32)
        pr = np.zeros(df.height, dtype=np.float32)
        for f in (0, 1):
            m = fold == f
            if m.any():
                pr[m] = lm[f].predict(X[m], **PRED_KW)
        if xm:
            Xx = df.select(xf).to_numpy().astype(np.float32)
            px = np.zeros(df.height, dtype=np.float32)
            for f in (0, 1):
                m = fold == f
                if m.any():
                    px[m] = xm[f].predict(xgb.DMatrix(Xx[m]))
            pr = (pr + px) / 2
        out.append(df.select("s1_row", "p_row").with_columns(pl.Series("prob", pr)))
    return pl.concat(out)


if __name__ == "__main__":
    s1, gt = load_gt_rows("trainsim")
    rows = s1["s1_row"].cast(pl.UInt32).to_numpy()
    for spec in sys.argv[1:]:
        lt, xt = (spec.split(":") + [None])[:2]
        o = score(lt, xt)
        f = evaluate(select_expected_f(one_owner(o, "prob", "q"), "q"), gt, rows)
        print(f"{spec}: stage-1 macro F0.5 on current trainsim = {f:.5f}", flush=True)
