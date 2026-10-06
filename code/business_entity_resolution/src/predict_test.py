"""Score every test candidate pair with the fold models (averaged) and write the submission."""
import glob
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from config import OUTPUT, WORK, norm_path
from decide import exclusive, one_owner, select_expected_f, select_threshold
from train_lgb import PRED_KW
from write_outputs import write_submission


def score_parts(split, tag, feats=None):
    models = [lgb.Booster(model_file=os.path.join(WORK, f"lgb_{tag}_fold{f}.txt")) for f in (0, 1)]
    if feats is None:
        with open(os.path.join(WORK, f"lgb_{tag}_features.json")) as f:
            feats = json.load(f)
    out = []
    for p in sorted(glob.glob(os.path.join(WORK, f"feat_{split}", "part_*.parquet"))):
        df = pl.read_parquet(p)
        X = df.select(feats).to_numpy().astype(np.float32)
        pr = np.mean([m.predict(X, **PRED_KW) for m in models], axis=0).astype(np.float32)
        out.append(df.select("s1_row", "p_row", "blk_cos").with_columns(pl.Series("prob", pr)))
    return pl.concat(out)


def main(tag="m1", strategy="expF", thr=0.5):
    t = time.time()
    path = os.path.join(WORK, f"test_scored_{tag}.parquet")
    if os.path.exists(os.path.join(WORK, f"lgb_{tag}_fold0.txt")) and tag.startswith("m"):
        scored = score_parts("test", tag)          # stage-1 models: score the feature parts
        scored.write_parquet(path)
    else:
        scored = pl.read_parquet(path)             # e.g. stage-2 scores written by train_stage2.py
    print(f"scored {scored.height} test pairs in {time.time() - t:.0f}s")
    s1_ids = pl.read_parquet(norm_path("test", 1), columns=["entity_id"])["entity_id"]
    pool_ids = pl.concat([pl.read_parquet(norm_path("test", 2), columns=["entity_id"]),
                          pl.read_parquet(norm_path("test", 3), columns=["entity_id"])])["entity_id"]
    if strategy.startswith("owner"):
        oo = one_owner(scored, "prob", "q")
        sel = select_expected_f(oo, "q") if strategy == "ownerF" else select_threshold(oo, thr, "q")
    else:
        ex = exclusive(scored, "prob")
        sel = select_expected_f(ex, "prob") if strategy == "expF" else select_threshold(ex, thr, "prob")
    # diagnostics per country
    s1c = pl.read_parquet(norm_path("test", 1), columns=["country"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
    diag = (s1c.join(sel.group_by("s1_row").len(), on="s1_row", how="left").fill_null(0)
               .group_by("country").agg(pl.col("len").mean().alias("pred_per_s1"), (pl.col("len") == 0).mean().alias("empty_frac")))
    print(diag)
    to_ids = lambda d: d.select(pl.Series("s1_id", s1_ids.gather(d["s1_row"].to_numpy())),
                                pl.Series("p_id", pool_ids.gather(d["p_row"].to_numpy())))
    write_submission(s1_ids, to_ids(sel), to_ids(scored), OUTPUT)
    print(f"predicted pairs: {sel.height} ({sel.height / len(s1_ids):.3f} per S1); "
          f"S1 with >=1 match: {sel['s1_row'].n_unique() / len(s1_ids):.4f}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["m1"]), *(sys.argv[2:3] or ["expF"]), *( [float(sys.argv[3])] if len(sys.argv) > 3 else []))
