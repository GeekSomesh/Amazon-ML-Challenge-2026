"""Where exactly is macro F0.5 lost? For an OOF file, attribute every false-negative and false-positive
pair of the final decision (one-owner + expected-F) to record subgroups, and compare mean model
probability with the true match rate per subgroup (calibration).

  python loss_by_group.py <tag>
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import glob
import os
import sys

import polars as pl

from config import WORK, norm_path
from decide import one_owner, select_expected_f
from eval_oof import load_gt_rows

COLS = ["s1_row", "p_row", "p_addr_empty", "p_is_domain", "p_name_unseen", "a_tset", "num_first_eq",
        "num_n1", "num_n2", "blk_rank_p", "n_tset"]


def main(tag):
    # pairs below 0.002 carry negligible one-owner mass (memory: keep them only if true, for FN accounting)
    oof = (pl.scan_parquet(os.path.join(WORK, f"oof_{tag}.parquet")).select("s1_row", "p_row", "label", "prob")
             .filter((pl.col("prob") >= 0.002) | (pl.col("label") == 1)).collect(engine="streaming"))
    s1, gt = load_gt_rows("trainsim")
    sel = select_expected_f(one_owner(oof, "prob", "q"), "q").select("s1_row", "p_row").with_columns(pl.lit(1).alias("sel"))
    d = oof.join(sel, on=["s1_row", "p_row"], how="left").with_columns(pl.col("sel").fill_null(0))
    d = d.filter((pl.col("prob") >= 0.02) | (pl.col("label") == 1))
    f = pl.concat([pl.read_parquet(p, columns=COLS).join(d.select("s1_row", "p_row"), on=["s1_row", "p_row"], how="semi")
                   for p in sorted(glob.glob(os.path.join(WORK, "feat_trainsim", "part_*.parquet")))])
    d = d.join(f, on=["s1_row", "p_row"], how="left")
    pid = pl.concat([pl.read_parquet(norm_path("trainsim", s), columns=["entity_id", "has_alias"]) for s in (2, 3)]).with_row_index("p_row")
    d = d.join(pid.with_columns(pl.col("p_row").cast(pl.UInt32)), on="p_row", how="left")
    both = (pl.col("num_n1") > 0) & (pl.col("num_n2") > 0)
    d = d.with_columns(
        pl.when(pl.col("entity_id").str.contains("DCY")).then(pl.lit("synthetic_decoy"))
          .when(pl.col("p_addr_empty") > 0).then(pl.lit("addr_empty"))
          .when(pl.col("has_alias")).then(pl.lit("alias"))
          .when(pl.col("p_is_domain") > 0).then(pl.lit("domain"))
          .when(pl.col("p_name_unseen") >= 0.99).then(pl.lit("name_unseen"))
          .when(both & (pl.col("num_first_eq") == 0)).then(pl.lit("num_mismatch"))
          .when(~both).then(pl.lit("num_missing"))
          .otherwise(pl.lit("regular")).alias("grp"))
    n_s1 = s1.height
    # the per-pair F0.5 impact is roughly equal; report counts and shares
    fn = d.filter((pl.col("label") == 1) & (pl.col("sel") == 0))
    fp = d.filter((pl.col("label") == 0) & (pl.col("sel") == 1))
    miss = gt.join(pl.scan_parquet(os.path.join(WORK, f"oof_{tag}.parquet")).select("s1_row", "p_row").collect(engine="streaming"), on=["s1_row", "p_row"], how="anti").height
    print(f"[{tag}] S1={n_s1}  FN pairs={fn.height} (+{miss} never candidates)  FP pairs={fp.height}", flush=True)
    pl.Config.set_tbl_rows(40)
    tab = (d.group_by("grp").agg(pl.len().alias("pairs"), pl.col("label").sum().alias("true_pairs"),
                                 pl.col("prob").mean().round(4).alias("mean_prob"), pl.col("label").mean().round(4).alias("true_rate"),
                                 ((pl.col("label") == 1) & (pl.col("sel") == 0)).sum().alias("FN"),
                                 ((pl.col("label") == 0) & (pl.col("sel") == 1)).sum().alias("FP"))
             .sort("FN", descending=True))
    print(tab, flush=True)
    # FN by probability bucket: lost because of low prob, or lost in the one-owner / expected-F step?
    print(fn.with_columns(pl.col("prob").cut([0.1, 0.3, 0.5, 0.7, 0.9]).alias("pb")).group_by("pb").len().sort("pb"), flush=True)
    print(fp.with_columns(pl.col("prob").cut([0.5, 0.7, 0.9, 0.97]).alias("pb")).group_by("pb").len().sort("pb"), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
