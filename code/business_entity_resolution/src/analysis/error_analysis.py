"""Where is macro F0.5 lost? Decomposes the per-entity loss of a decision on OOF predictions."""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import os
import sys

import polars as pl

from config import WORK, norm_path
from decide import one_owner, select_threshold
from eval_oof import load_gt_rows
from metric import f05_macro


def main(tag="m1", thr=0.6):
    oof = pl.read_parquet(os.path.join(WORK, f"oof_{tag}.parquet"), columns=["s1_row", "p_row", "label", "prob"])
    s1, gt = load_gt_rows("trainsim")
    sel = select_threshold(one_owner(oof, "prob", "q"), thr, "q")
    rows = s1["s1_row"].cast(pl.UInt32)
    d = f05_macro(sel.rename({"s1_row": "s1", "p_row": "p"}), gt.rename({"s1_row": "s1", "p_row": "p"}),
                  rows.rename("s1"), return_per_entity=True)
    d = d.with_columns((1 - pl.col("f")).alias("loss"),
                       pl.when((pl.col("ngt") == 0) & (pl.col("npred") > 0)).then(pl.lit("singleton_false_merge"))
                         .when((pl.col("ngt") > 0) & (pl.col("npred") == 0)).then(pl.lit("missed_entity_entirely"))
                         .when((pl.col("ngt") > 0) & (pl.col("tp") == 0)).then(pl.lit("all_predictions_wrong"))
                         .when(pl.col("npred") > pl.col("tp")).then(pl.lit("has_false_positive"))
                         .when(pl.col("tp") < pl.col("ngt")).then(pl.lit("partial_recall_only"))
                         .otherwise(pl.lit("perfect")).alias("cat"))
    n = d.height
    print(f"macro F0.5 = {d['f'].mean():.5f}; total loss = {d['loss'].sum() / n:.5f}")
    print(d.group_by("cat").agg(pl.len().alias("entities"), (pl.col("loss").sum() / n).alias("loss_share")).sort("loss_share", descending=True))
    # false positives: is the predicted pool record an orphan (its entity is not in this S1 table)?
    fp = sel.join(gt, on=["s1_row", "p_row"], how="anti")
    owned = gt.select("p_row").unique().with_columns(pl.lit(True).alias("owned"))
    fp = fp.join(owned, on="p_row", how="left").with_columns(pl.col("owned").fill_null(False))
    print(f"false-positive pairs {fp.height}: orphan pool records {(~fp['owned']).sum()}, belong to another S1 {fp['owned'].sum()}")
    fn = gt.join(sel, on=["s1_row", "p_row"], how="anti").join(oof.select("s1_row", "p_row", "prob"), on=["s1_row", "p_row"], how="left")
    print(f"false-negative pairs {fn.height}: not in candidates {fn['prob'].is_null().sum()}, "
          f"prob<0.1 {(fn['prob'] < 0.1).sum()}, 0.1-0.6 {((fn['prob'] >= 0.1) & (fn['prob'] < 0.6)).sum()}, >=0.6 (lost to one-owner) {(fn['prob'] >= 0.6).sum()}")
    d.write_parquet(os.path.join(WORK, f"per_entity_{tag}.parquet"))
    fp.write_parquet(os.path.join(WORK, f"fp_{tag}.parquet"))
    fn.write_parquet(os.path.join(WORK, f"fn_{tag}.parquet"))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["m1"]))
