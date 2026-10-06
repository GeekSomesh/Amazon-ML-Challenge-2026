"""S1 entities with exactly ONE predicted match: how often are they true singletons (validation), how common are
they per country on test, and what do the French ones look like (source, number relation, name relation, q)?
  python one_match.py <oof_tag> <test_tag>"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import os
import sys

import polars as pl

from config import WORK, norm_path
from decide import one_owner, select_expected_f


def sel_q(scored):
    oo = one_owner(scored.select("s1_row", "p_row", "prob"), "prob", "q")
    return select_expected_f(oo, "q").join(oo.select("s1_row", "p_row", "q"), on=["s1_row", "p_row"])


def main(oof_tag, test_tag):
    from eval_oof import load_gt_rows
    s1, gt = load_gt_rows("trainsim")
    oof = pl.scan_parquet(os.path.join(WORK, f"oof_{oof_tag}.parquet")).select("s1_row", "p_row", "prob").filter(pl.col("prob") >= 0.002).collect(engine="streaming")
    sel = sel_q(oof)
    k = sel.group_by("s1_row").agg(pl.len().alias("k"), pl.col("q").max().alias("qmax"))
    ngt = gt.group_by("s1_row").len().rename({"len": "ngt"})
    k = k.join(ngt, on="s1_row", how="left").with_columns(pl.col("ngt").fill_null(0))
    one = k.filter(pl.col("k") == 1).join(sel, on="s1_row").join(gt.with_columns(pl.lit(1).alias("y")), on=["s1_row", "p_row"], how="left").with_columns(pl.col("y").fill_null(0))
    print("VALIDATION S1 with exactly 1 predicted match:", one.height, f"({one.height / s1.height:.4f} of S1)",
          " singleton share:", round((one["ngt"] == 0).mean(), 4), " match correct:", round(one["y"].mean(), 4))
    print(one.with_columns(pl.col("q").cut([0.7, 0.9, 0.97, 0.99]).alias("qb")).group_by("qb").agg(
        pl.len(), (pl.col("ngt") == 0).mean().round(3).alias("singleton"), pl.col("y").mean().round(3).alias("correct")).sort("qb"))
    te = pl.read_parquet(os.path.join(WORK, f"test_scored_{test_tag}.parquet"), columns=["s1_row", "p_row", "prob"])
    st = sel_q(te)
    kt = st.group_by("s1_row").agg(pl.len().alias("k"))
    c = pl.read_parquet(norm_path("test", 1), columns=["country"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
    dist = c.join(kt, on="s1_row", how="left").with_columns(pl.col("k").fill_null(0).clip(0, 4))
    print("TEST share of S1 by predicted count (0..4+):")
    print(dist.group_by("country", "k").len().join(c.group_by("country").len().rename({"len": "N"}), on="country")
          .with_columns((pl.col("len") / pl.col("N")).round(4).alias("share")).pivot(on="country", index="k", values="share").sort("k"))
    onet = kt.filter(pl.col("k") == 1).join(st, on="s1_row").join(c, on="s1_row")
    print(onet.with_columns(pl.col("q").cut([0.7, 0.9, 0.97, 0.99]).alias("qb")).group_by("country", "qb").len().sort("country", "qb")
          .pivot(on="country", index="qb", values="len"))
    onet.select("s1_row", "p_row", "q", "country").write_parquet(os.path.join(WORK, "one_match_test.parquet"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
