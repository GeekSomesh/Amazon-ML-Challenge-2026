"""Label-free comparison of two scored test files: how the final selections differ by pair type
(house number shifted with the same digit count / digit-count change / same number / no numbers)."""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import glob
import os
import sys

import polars as pl

from config import WORK, norm_path
from decide import one_owner, select_expected_f


def selections(tag):
    sc = pl.read_parquet(os.path.join(WORK, f"test_scored_{tag}.parquet"), columns=["s1_row", "p_row", "prob"])
    return select_expected_f(one_owner(sc, "prob", "q"), "q").with_columns(pl.lit(1).alias(tag))


def main(a, b):
    sa, sb = selections(a), selections(b)
    u = sa.join(sb, on=["s1_row", "p_row"], how="full", coalesce=True).fill_null(0)
    cols = ["s1_row", "p_row", "num_first_in", "num_lendiff", "num_n1", "num_n2"]
    f = pl.concat([pl.read_parquet(p, columns=cols).join(u.select("s1_row", "p_row"), on=["s1_row", "p_row"], how="semi")
                   for p in sorted(glob.glob(os.path.join(WORK, "feat_test", "part_*.parquet")))])
    s1 = pl.read_parquet(norm_path("test", 1), columns=["country"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
    d = u.join(f, on=["s1_row", "p_row"]).join(s1, on="s1_row").with_columns(
        pl.when((pl.col("num_n1") == 0) | (pl.col("num_n2") == 0)).then(pl.lit("no numbers"))
          .when(pl.col("num_first_in") == 1).then(pl.lit("same number"))
          .when(pl.col("num_lendiff") == 0).then(pl.lit("shifted number (same length)"))
          .otherwise(pl.lit("digit-count change")).alias("type"))
    print(d.group_by("country", "type").agg(pl.col(a).sum().alias(a), pl.col(b).sum().alias(b)).sort("country", "type"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
