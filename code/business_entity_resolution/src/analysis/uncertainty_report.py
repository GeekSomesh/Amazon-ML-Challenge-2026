"""Label-free diagnostic: how confident is the matcher per country on a scored split?
Compares the share of selected pairs that are borderline, e.g. to see whether an unseen
country (France) behaves like the countries seen in training."""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import os
import sys

import polars as pl

from config import WORK, norm_path
from decide import one_owner


def report(scored: pl.DataFrame, split: str, name: str):
    s1 = pl.read_parquet(norm_path(split, 1), columns=["country"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
    oo = one_owner(scored, "prob", "q").join(s1, on="s1_row")
    r = oo.group_by("country").agg(
        (pl.col("q") >= 0.6).sum().alias("sel"),
        ((pl.col("q") >= 0.2) & (pl.col("q") < 0.6)).sum().alias("below"),
        ((pl.col("q") >= 0.6) & (pl.col("q") < 0.9)).sum().alias("above"),
        (pl.col("q") >= 0.97).sum().alias("sure"))
    r = r.with_columns((pl.col("below") / pl.col("sel")).round(4).alias("just_below"),
                       (pl.col("above") / pl.col("sel")).round(4).alias("just_above"),
                       (pl.col("sure") / pl.col("sel")).round(4).alias("very_sure")).sort("country")
    print(name)
    print(r.select("country", "sel", "just_below", "just_above", "very_sure"))


if __name__ == "__main__":
    tag = sys.argv[1]
    if os.path.exists(os.path.join(WORK, f"oof_{tag}.parquet")):
        report(pl.read_parquet(os.path.join(WORK, f"oof_{tag}.parquet"), columns=["s1_row", "p_row", "prob"]), "trainsim", f"VALIDATION {tag}")
    if os.path.exists(os.path.join(WORK, f"test_scored_{tag}.parquet")):
        report(pl.read_parquet(os.path.join(WORK, f"test_scored_{tag}.parquet"), columns=["s1_row", "p_row", "prob"]), "test", f"TEST {tag}")
