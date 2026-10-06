"""France: candidate pairs whose names differ ONLY by injected French words (participations, holding, distribution,
international, developpement, groupe, fils, france, associes) and legal forms, with the S1 house number present in
the pool address -> how many, current probabilities, selection status under the decision layer.

  python fr_marker_pairs.py [scored_tag]   (default: s12ar, the final recalibrated stage-2 scores)"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import os
import sys

import polars as pl

from config import WORK
from decide import one_owner, select_expected_f

INJ = ["participations", "holding", "distribution", "international", "developpement", "groupe", "fils", "france", "associes"]
LEGAL = ["sarl", "sas", "sasu", "sa", "sci", "eurl", "snc", "ei", "eirl", "cie", "co", "and", "et", "the", "de", "du", "des", "la", "le", "les"]


def core(e):
    t = (e.str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase().str.replace_all(r"\b([a-z])\.", "$1")
          .str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars().str.split(" "))
    return t.list.eval(pl.element().filter(~pl.element().is_in(INJ + LEGAL) & (pl.element() != ""))).list.sort().list.join(" ")


def main(tag="s12ar"):
    t1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet")).with_row_index("s1_row").filter(pl.col("country") == "France")
    sc = pl.read_parquet(os.path.join(WORK, f"test_scored_{tag}.parquet"), columns=["s1_row", "p_row", "prob"])
    oo = one_owner(sc, "prob", "q")
    sel = select_expected_f(oo, "q").with_columns(pl.lit(1).alias("sel"))
    d = oo.filter(pl.col("s1_row").is_in(t1["s1_row"].cast(pl.UInt32).implode()) & (pl.col("prob") >= 0.0005))
    n2 = pl.scan_parquet(os.path.join(WORK, "test_s2.parquet")).select(pl.len()).collect().item()
    pool = pl.concat([pl.scan_parquet(os.path.join(WORK, f"test_s{s}.parquet")).select("business_name", "business_address")
                      .with_row_index("p_row", offset=o).filter(pl.col("p_row").is_in(d["p_row"].implode())).collect() for s, o in ((2, 0), (3, n2))])
    d = (d.join(t1.select(pl.col("s1_row").cast(pl.UInt32), "business_name", "business_address"), on="s1_row")
          .join(pool.with_columns(pl.col("p_row").cast(pl.UInt32)), on="p_row", suffix="_p")
          .join(sel, on=["s1_row", "p_row"], how="left").with_columns(pl.col("sel").fill_null(0)))
    num = lambda c: pl.col(c).str.replace_all(r"(\d)\s*-\s*(\d)", "$1$2").str.extract(r"(\d+)", 1).cast(pl.Int64, strict=False)
    inj = r"(?i)\b(participations|holding|distribution|international|d[eé]veloppement|groupe)\b"
    d = d.with_columns(core(pl.col("business_name")).alias("c1"), core(pl.col("business_name_p")).alias("c2"),
                       num("business_address").alias("n1"), num("business_address_p").alias("n2"),
                       (pl.col("business_name_p").str.contains(inj) & ~pl.col("business_name").str.contains(inj)).alias("pool_inj"))
    d = d.with_columns((pl.col("c1") == pl.col("c2")).alias("same_core"),
                       pl.when(pl.col("n1").is_null() | pl.col("n2").is_null()).then(pl.lit("missing"))
                         .when(pl.col("n1") == pl.col("n2")).then(pl.lit("eq")).otherwise(pl.lit("diff")).alias("num"))
    x = d.filter(pl.col("same_core") & pl.col("pool_inj"))
    pl.Config.set_tbl_rows(20)
    print("France pairs: names equal after removing injected words, pool adds groupe/holding/...:")
    print(x.group_by("num", "sel").agg(pl.len(), pl.col("q").mean().round(3).alias("mean_q"), pl.col("prob").mean().round(3).alias("mean_p"),
                                       (pl.col("q") >= 0.3).sum().alias("q>=0.3")).sort("num", "sel"))
    y = x.filter((pl.col("num") == "eq") & (pl.col("sel") == 0))
    print("unselected same-number examples:")
    for r in y.sample(min(12, y.height), seed=3).iter_rows(named=True):
        print(f"  q={r['q']:.3f} p={r['prob']:.3f} | {r['business_name']!r} {r['business_address']!r} -> {r['business_name_p']!r} {r['business_address_p']!r}")
    x.select("s1_row", "p_row", "num", "sel", "q", "prob").write_parquet(os.path.join(WORK, "fr_marker_pairs.parquet"))


if __name__ == "__main__":
    main(*sys.argv[1:2])
