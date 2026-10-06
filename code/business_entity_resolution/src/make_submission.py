"""Final decision layer on a scored test file -> output/matching_results.tsv + candidate_pairs.tsv.

  python make_submission.py <tag> [strategy]     strategy: ownerF (default) | owner:<thr>
  one-owner renormalisation -> expected-F0.5 subset per S1 -> English decoy-marker post-filter
"""
import os
import sys

import polars as pl

from config import OUTPUT, WORK, norm_path
from decide import one_owner, select_expected_f, select_threshold
from postfilter import MARKERS_TRAIN, drop_marker_pairs
from write_outputs import write_submission


def final_candidates(sc: pl.DataFrame, stage1_tag):
    """candidate_pairs.tsv = exactly the pairs the final (stage-2) model runs inference over:
    the stage-1 matcher acts as a learned blocker and passes on pairs with prob >= MIN_P1."""
    if stage1_tag is None:
        return sc.select("s1_row", "p_row")
    from stage2 import MIN_P1
    s1p = pl.read_parquet(os.path.join(WORK, f"test_scored_{stage1_tag}.parquet"), columns=["s1_row", "p_row", "prob"])
    return s1p.filter(pl.col("prob") >= MIN_P1).select("s1_row", "p_row")


def main(tag, strategy="ownerF", stage1_tag=None):
    sc = pl.read_parquet(os.path.join(WORK, f"test_scored_{tag}.parquet"), columns=["s1_row", "p_row", "prob"])
    if stage1_tag is not None:
        # only pairs that reached the final model can be matched
        sc = sc.join(final_candidates(sc, stage1_tag), on=["s1_row", "p_row"], how="semi")
    t1 = pl.read_parquet(norm_path("test", 1), columns=["entity_id", "name_full", "country"])
    tp = pl.concat([pl.read_parquet(norm_path("test", 2), columns=["entity_id", "name_full"]),
                    pl.read_parquet(norm_path("test", 3), columns=["entity_id", "name_full"])])
    oo = one_owner(sc, "prob", "q")
    sel = select_expected_f(oo, "q") if strategy == "ownerF" else select_threshold(oo, float(strategy.split(":")[1]), "q")
    sel = drop_marker_pairs(sel, t1["name_full"], tp["name_full"], MARKERS_TRAIN)
    ids = lambda d: d.select(pl.Series("s1_id", t1["entity_id"].gather(d["s1_row"].to_numpy())),
                             pl.Series("p_id", tp["entity_id"].gather(d["p_row"].to_numpy())))
    write_submission(t1["entity_id"], ids(sel), ids(sc), OUTPUT)
    per = (t1.with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
             .join(sel.group_by("s1_row").len(), on="s1_row", how="left").fill_null(0)
             .group_by("country").agg(pl.col("len").mean().round(3).alias("pred_per_s1"), (pl.col("len") == 0).mean().round(4).alias("empty_frac")))
    print(per.sort("country"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
