"""Leave-one-country-out check: how well does the matcher transfer to a country it never
saw? (proxy for France, which only exists in the test set)."""
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
from decide import evaluate, exclusive, select_expected_f
from eval_oof import load_gt_rows
from train_lgb import PARAMS, feature_names, parts


def main(train_c="US", test_c="India", frac=0.12):
    s1c = pl.read_parquet(norm_path("trainsim", 1), columns=["country"]).with_row_index("s1_row")
    rows_tr = s1c.filter(pl.col("country") == train_c)["s1_row"].cast(pl.UInt32)
    rows_te = s1c.filter(pl.col("country") == test_c)["s1_row"].cast(pl.UInt32)
    feats = feature_names("trainsim")
    lf = pl.scan_parquet(parts("trainsim"))
    tr = (lf.filter(pl.col("s1_row").is_in(rows_tr.implode()) & ((pl.col("s1_row").hash(seed=3) % 1000) < int(frac * 1000)))
            .collect())
    te = lf.filter(pl.col("s1_row").is_in(rows_te.implode())).select(["s1_row", "p_row", "label"] + feats).collect()
    m = lgb.train(PARAMS, lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32), label=tr["label"].to_numpy()), 400)
    te = te.select("s1_row", "p_row", "label").with_columns(pl.Series("prob", m.predict(te.select(feats).to_numpy().astype(np.float32))))
    s1, gt = load_gt_rows("trainsim")
    r = rows_te.to_numpy()
    f = evaluate(select_expected_f(exclusive(te, "prob"), "prob"), gt.filter(pl.col("s1_row").is_in(rows_te.implode())), r)
    print(f"train on {train_c} ({tr.height} pairs) -> {test_c}: macro F0.5 = {f:.5f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
