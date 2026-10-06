"""Label-free noise-rate recalibration for country labels that never occur in training.

The matcher learned from US / India how often a TRUE copy of a record carries a different house
number (their noise process drops / changes digits: among confident matches the S1 house number is
missing from 6-12% of pool records). For a new country we estimate the same rate from its own
confident matches (France: ~0.9%); when the rate is lower,
a number mismatch is correspondingly stronger evidence of a different business (decoy branch at a
nearby number), so the odds of mismatch pairs are rescaled by r_country / r_train (likelihood-ratio
correction, clipped to [0.05, 1]). Countries seen in training are left untouched.

  python recalibrate.py <test_tag> [oof_tag]   test_scored_<test_tag> + oof_<oof_tag> -> test_scored_<test_tag>r
"""
import glob
import os
import sys

import numpy as np
import polars as pl

from config import WORK, norm_path

CONF = 0.97


def _mismatch_frame(split, scored):
    """(s1_row, p_row, prob, mismatch, both_have_numbers) for all scored pairs of a split."""
    cols = ["s1_row", "p_row", "num_first_in", "num_n1", "num_n2"]
    f = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob(os.path.join(WORK, f"feat_{split}", "part_*.parquet")))])
    d = scored.join(f, on=["s1_row", "p_row"], how="left")
    return d.with_columns(((pl.col("num_n1") > 0) & (pl.col("num_n2") > 0)).alias("_both"),
                          (pl.col("num_first_in") == 0).alias("_mis")).drop("num_first_in", "num_n1", "num_n2")


def mismatch_rate(d: pl.DataFrame, country: pl.DataFrame):
    """rate of 'S1 house number absent from the pool record' among confident pairs, per country."""
    x = d.filter((pl.col("prob") >= CONF) & pl.col("_both")).join(country, on="s1_row")
    return x.group_by("country").agg(pl.col("_mis").mean().alias("rate"), pl.len().alias("n"))


def recalibrate(test_scored: pl.DataFrame, train_oof: pl.DataFrame, min_factor=0.05):
    ctest = pl.read_parquet(norm_path("test", 1), columns=["country"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
    ctrain = pl.read_parquet(norm_path("trainsim", 1), columns=["country"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
    dtr = _mismatch_frame("trainsim", train_oof)
    rtr = mismatch_rate(dtr, ctrain)
    r_train = float((rtr["rate"] * rtr["n"]).sum() / rtr["n"].sum())
    seen = set(rtr["country"].to_list())
    dte = _mismatch_frame("test", test_scored)
    rte = mismatch_rate(dte, ctest)
    factors = {c: float(np.clip(r / r_train, min_factor, 1.0)) for c, r in zip(rte["country"], rte["rate"]) if c not in seen}
    print(f"train mismatch rate {r_train:.4f}; test rates {dict(zip(rte['country'].to_list(), [round(x, 4) for x in rte['rate'].to_list()]))}; "
          f"factors for unseen countries {factors}", flush=True)
    if not factors:
        return test_scored
    fac = pl.DataFrame({"country": list(factors), "_fac": list(factors.values())})
    d = dte.join(ctest, on="s1_row").join(fac, on="country", how="left").with_columns(pl.col("_fac").fill_null(1.0))
    p = pl.col("prob").clip(1e-6, 1 - 1e-6)
    odds = p / (1 - p) * pl.when(pl.col("_both") & pl.col("_mis")).then(pl.col("_fac")).otherwise(1.0)
    d = d.with_columns((odds / (1 + odds)).cast(pl.Float32).alias("prob"))
    return d.select(test_scored.columns)


if __name__ == "__main__":
    test_tag = sys.argv[1]
    oof_tag = sys.argv[2] if len(sys.argv) > 2 else test_tag
    scored = pl.read_parquet(os.path.join(WORK, f"test_scored_{test_tag}.parquet"))
    oof = pl.read_parquet(os.path.join(WORK, f"oof_{oof_tag}.parquet"), columns=["s1_row", "p_row", "prob"])
    recalibrate(scored, oof).write_parquet(os.path.join(WORK, f"test_scored_{test_tag}r.parquet"))
    print(f"wrote test_scored_{test_tag}r.parquet", flush=True)
