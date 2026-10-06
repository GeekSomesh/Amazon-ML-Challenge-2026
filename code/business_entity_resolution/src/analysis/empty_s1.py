"""S1 entities the decision predicts EMPTY: what do their best candidates look like (per country)?
Validation: how many empty-predicted S1 are true singletons. Test: best-candidate profile + raw examples.
  python empty_s1.py <oof_tag> <test_tag> <country> [n_examples]"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import os
import sys

import polars as pl

from config import WORK, norm_path
from decide import one_owner, select_expected_f


def empties(scored):
    oo = one_owner(scored.select("s1_row", "p_row", "prob"), "prob", "q")
    sel = select_expected_f(oo, "q")
    best = oo.sort("q", descending=True).group_by("s1_row").first().rename({"p_row": "best_p", "q": "best_q", "prob": "best_prob"})
    return oo, sel, best


def main(oof_tag, test_tag, country, n_ex="12"):
    from eval_oof import load_gt_rows
    s1, gt = load_gt_rows("trainsim")
    oof = pl.scan_parquet(os.path.join(WORK, f"oof_{oof_tag}.parquet")).select("s1_row", "p_row", "prob").filter(pl.col("prob") >= 0.002).collect(engine="streaming")
    oo, sel, best = empties(oof)
    rows = s1.select(pl.col("s1_row").cast(pl.UInt32), "country")
    emp = rows.join(sel.select("s1_row").unique(), on="s1_row", how="anti").join(best, on="s1_row", how="left")
    ngt = gt.group_by("s1_row").len().rename({"len": "ngt"})
    emp = emp.join(ngt, on="s1_row", how="left").with_columns(pl.col("ngt").fill_null(0))
    print("VALIDATION empty-predicted S1:", emp.height, f"({emp.height / rows.height:.4f})", " true singletons:", round((emp['ngt'] == 0).mean(), 4))
    print(emp.with_columns(pl.col("best_q").fill_null(0).cut([0.05, 0.2, 0.4]).alias("bq")).group_by("bq").agg(
        pl.len(), (pl.col("ngt") == 0).mean().round(3).alias("singleton_rate")).sort("bq"))
    te = pl.read_parquet(os.path.join(WORK, f"test_scored_{test_tag}.parquet"), columns=["s1_row", "p_row", "prob"])
    oo, sel, best = empties(te)
    t1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet")).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
    emp = t1.join(sel.select("s1_row").unique(), on="s1_row", how="anti").join(best, on="s1_row", how="left")
    print("TEST empty-predicted S1 by country:", emp.group_by("country").len().join(t1.group_by("country").len().rename({"len": "S1"}), on="country")
          .with_columns((pl.col("len") / pl.col("S1")).round(4).alias("rate")).sort("country").rows())
    print(emp.with_columns(pl.col("best_q").fill_null(0).cut([0.05, 0.2, 0.4]).alias("bq")).group_by("country", "bq").len().sort("country", "bq")
          .pivot(on="country", index="bq", values="len"))
    ex = emp.filter((pl.col("country") == country) & pl.col("best_q").is_not_null()).sample(int(n_ex), seed=5)
    n2 = pl.scan_parquet(os.path.join(WORK, "test_s2.parquet")).select(pl.len()).collect().item()
    rp = pl.concat([pl.scan_parquet(os.path.join(WORK, f"test_s{s}.parquet")).select("entity_id", "business_name", "business_address")
                    .with_row_index("best_p", offset=o).filter(pl.col("best_p").is_in(ex["best_p"].implode())).collect() for s, o in ((2, 0), (3, n2))])
    ex = ex.join(rp.with_columns(pl.col("best_p").cast(pl.UInt32)), on="best_p", suffix="_p")
    for r in ex.sort("best_q", descending=True).iter_rows(named=True):
        print(f"  best q={r['best_q']:.3f} | {r['business_name']!r} {r['business_address']!r}\n        -> {r['entity_id_p'][:2]} {r['business_name_p']!r} {r['business_address_p']!r}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
