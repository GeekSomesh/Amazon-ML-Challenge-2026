"""Same-address category-word swaps: a content word of the S1 name is replaced by a DIFFERENT content word
(not a typo, not a filler / legal form). Label-free check of how the model scores them per country (test)
and whether it is calibrated for them where labels exist (trainsim OOF).

  python swap_calib.py <oof_tag> <test_tag>
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import glob
import os
import sys

import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein

from config import WORK, norm_path
from text_norm import FR_FILLER, NAME_STOP

IGNORE = NAME_STOP | FR_FILLER | {"partners", "group", "groupe", "cie", "co"}


def swap_flag(n1, n2):
    out = np.zeros(len(n1), dtype=bool)
    for i, (a, b) in enumerate(zip(n1, n2)):
        x = set(a.split()) - IGNORE
        y = set(b.split()) - IGNORE
        if not (x & y):
            continue
        da, db = x - y, y - x
        if da and db and not all(any(Levenshtein.normalized_similarity(t, u) >= 0.75 for u in da) for t in db):
            out[i] = True
    return out


def frame(split, scored):
    F = ["s1_row", "p_row", "a_tset", "num_first_eq"]
    f = pl.concat([pl.read_parquet(p, columns=F).filter((pl.col("a_tset") >= 97) & (pl.col("num_first_eq") > 0))
                   .join(scored.select("s1_row", "p_row"), on=["s1_row", "p_row"], how="semi")
                   for p in sorted(glob.glob(os.path.join(WORK, f"feat_{split}", "part_*.parquet")))])
    d = f.join(scored, on=["s1_row", "p_row"])
    s1 = pl.read_parquet(norm_path(split, 1), columns=["country", "name_core"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
    n2 = pl.scan_parquet(norm_path(split, 2)).select(pl.len()).collect().item()
    pool = pl.concat([pl.scan_parquet(norm_path(split, s)).select("name_core").with_row_index("p_row", offset=o)
                      .filter(pl.col("p_row").is_in(d["p_row"].implode())).collect(engine="streaming") for s, o in ((2, 0), (3, n2))])
    d = d.join(s1, on="s1_row").join(pool.with_columns(pl.col("p_row").cast(pl.UInt32)).rename({"name_core": "nc2"}), on="p_row")
    return d.with_columns(pl.Series("swap", swap_flag(d["name_core"].to_list(), d["nc2"].to_list())))


def main(oof_tag, test_tag):
    tr = pl.scan_parquet(os.path.join(WORK, f"oof_{oof_tag}.parquet")).select("s1_row", "p_row", "prob", "label").filter(pl.col("prob") >= 0.05).collect(engine="streaming")
    d = frame("trainsim", tr).filter(pl.col("swap"))
    print("VALIDATION same-address swaps (prob>=0.05):")
    print(d.with_columns(pl.col("prob").cut([0.3, 0.5, 0.7, 0.9]).alias("pb")).group_by("country", "pb")
           .agg(pl.len(), pl.col("prob").mean().round(3).alias("mean_p"), pl.col("label").mean().round(3).alias("true_rate")).sort("country", "pb"))
    te = pl.read_parquet(os.path.join(WORK, f"test_scored_{test_tag}.parquet"), columns=["s1_row", "p_row", "prob"]).filter(pl.col("prob") >= 0.05)
    t = frame("test", te).filter(pl.col("swap"))
    n1 = pl.read_parquet(norm_path("test", 1), columns=["country"]).group_by("country").len().rename({"len": "S1"})
    print("TEST same-address swaps (prob>=0.05): count per S1 and mean prob")
    print(t.group_by("country").agg(pl.len(), pl.col("prob").mean().round(3).alias("mean_p"), (pl.col("prob") > 0.5).sum().alias("n_p>0.5"))
           .join(n1, on="country").with_columns((pl.col("n_p>0.5") / pl.col("S1")).round(4).alias("p>0.5_per_S1")).sort("country"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
