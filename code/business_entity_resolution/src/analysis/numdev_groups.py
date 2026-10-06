"""For an S1, look at its candidate pool records that have the same core name but a house number different from the
S1's ('deviating' records). Group them by (source, number). True base-number noise: deviating numbers differ between
S2 and S3; planted look-alike groups: the same shifted number, often in both sources.
Validation (labels): true rate by pattern. Test France: counts by pattern for pairs the France recalibration suppresses.
  python numdev_groups.py <oof_tag> <test_tag_no_recal> <test_tag_recal>"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import glob
import os
import sys

import polars as pl

from config import WORK, norm_path
from decide import one_owner, select_expected_f

F = ["s1_row", "p_row", "n_tset", "num_first_eq", "num_n1", "num_n2", "a_tset"]


def frame(split, scored, extra_cols=()):
    f = pl.concat([pl.read_parquet(p, columns=F).filter((pl.col("n_tset") >= 90) & (pl.col("num_n1") > 0) & (pl.col("num_n2") > 0))
                   .join(scored.select("s1_row", "p_row"), on=["s1_row", "p_row"], how="semi")
                   for p in sorted(glob.glob(os.path.join(WORK, f"feat_{split}", "part_*.parquet")))])
    d = f.join(scored, on=["s1_row", "p_row"])
    addr = pl.concat([pl.read_parquet(norm_path(split, s), columns=["addr_full"]) for s in (2, 3)]).with_row_index("p_row").with_columns(
        pl.col("p_row").cast(pl.UInt32), pl.col("addr_full").str.extract(r"\b(\d+)\b", 1).alias("pnum"))
    n2 = pl.scan_parquet(norm_path(split, 2)).select(pl.len()).collect().item()
    d = d.join(addr.select("p_row", "pnum"), on="p_row").with_columns(pl.when(pl.col("p_row") < n2).then(pl.lit("S2")).otherwise(pl.lit("S3")).alias("src"))
    dev = d.filter((pl.col("num_first_eq") == 0) & (pl.col("a_tset") >= 60))
    # per S1: deviating numbers seen in S2 and in S3
    g = dev.group_by("s1_row").agg(pl.col("pnum").filter(pl.col("src") == "S2").unique().alias("n_s2"),
                                   pl.col("pnum").filter(pl.col("src") == "S3").unique().alias("n_s3"))
    g = g.with_columns((pl.col("n_s2").list.set_intersection("n_s3").list.len() > 0).alias("same_num_both_src"),
                       ((pl.col("n_s2").list.len() > 0) & (pl.col("n_s3").list.len() > 0)).alias("both_src"))
    same = d.filter(pl.col("num_first_eq") > 0).group_by("s1_row").len().rename({"len": "n_same_num"})
    dev = dev.join(g, on="s1_row").join(same, on="s1_row", how="left").with_columns(pl.col("n_same_num").fill_null(0))
    return dev.with_columns(pl.when(pl.col("n_same_num") > 0).then(pl.lit("S1 also has same-number copies"))
                              .when(pl.col("same_num_both_src")).then(pl.lit("no same-number; SAME dev number in S2&S3"))
                              .when(pl.col("both_src")).then(pl.lit("no same-number; DIFFERENT dev numbers S2 vs S3"))
                              .otherwise(pl.lit("no same-number; one source only")).alias("pattern"))


def main(oof_tag, te_raw, te_recal):
    oof = pl.scan_parquet(os.path.join(WORK, f"oof_{oof_tag}.parquet")).select("s1_row", "p_row", "prob", "label").filter(pl.col("prob") >= 0.01).collect(engine="streaming")
    v = frame("trainsim", oof)
    pl.Config.set_tbl_rows(20)
    print("VALIDATION (US+India) same-name deviating-number candidates:")
    print(v.group_by("pattern").agg(pl.len(), pl.col("label").mean().round(3).alias("true_rate"), pl.col("prob").mean().round(3).alias("mean_p")).sort("pattern"))
    t_raw = pl.read_parquet(os.path.join(WORK, f"test_scored_{te_raw}.parquet"), columns=["s1_row", "p_row", "prob"]).filter(pl.col("prob") >= 0.01)
    t_rc = pl.read_parquet(os.path.join(WORK, f"test_scored_{te_recal}.parquet"), columns=["s1_row", "p_row", pl.col("prob").alias("prob_recal")] if False else ["s1_row", "p_row", "prob"]).rename({"prob": "prob_recal"})
    c = pl.read_parquet(norm_path("test", 1), columns=["country"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
    t = frame("test", t_raw).join(t_rc, on=["s1_row", "p_row"], how="left").join(c, on="s1_row")
    print("TEST same-name deviating-number candidates (prob without recal >= 0.01):")
    print(t.group_by("country", "pattern").agg(pl.len(), pl.col("prob").mean().round(3).alias("p_raw"), pl.col("prob_recal").mean().round(3).alias("p_recal"))
           .sort("country", "pattern"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
