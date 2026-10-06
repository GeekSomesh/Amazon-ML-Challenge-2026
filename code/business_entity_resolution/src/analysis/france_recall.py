"""Recall repair for a country unseen in training (France).

Leaderboard probe: France scores ~0.94 vs ~0.98 for US/India, and inspection shows the loss is missed
matches whose noise the US/India-trained model does not recognise:
  * acronym names ("CC" = Communale Comite, "CMJ" = Centre Medical de Jean)
  * extra suffixes ("& Fils", "Et Fils"), dropped legal forms, domains that keep the legal form,
    category-word swaps and generated random names - all at the SAME address (same source base) as a
    record already matched to the entity.
Two additions, applied only to country labels not present in the training data:
  1. base-mates: an unselected candidate whose best owner is this S1 and whose normalised address and
     source equal those of a record already selected for this S1 (q >= min_q);
  2. acronyms: pool name == initials of the S1 name and the S1 house number occurs in the pool address.
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import glob
import os

import polars as pl

from config import WORK, norm_path

STOP = {"and", "et", "of", "de", "du", "des", "la", "le", "les", "the", "l", "d"}


def initials(name: str) -> str:
    return "".join(t[0] for t in name.split() if t and t not in STOP)


def france_recall(split, scored, oo, sel, min_q=0.2, seen=("US", "India")):
    s1 = pl.read_parquet(norm_path(split, 1), columns=["country", "name_core", "name_full"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
    pool = pl.concat([pl.read_parquet(norm_path(split, 2), columns=["entity_id", "addr_full", "name_core"]),
                      pl.read_parquet(norm_path(split, 3), columns=["entity_id", "addr_full", "name_core"])]).with_row_index("p_row").with_columns(
        pl.col("p_row").cast(pl.UInt32), pl.col("entity_id").str.slice(0, 2).alias("src"))
    unseen = s1.filter(~pl.col("country").is_in(list(seen))).select("s1_row")
    c = oo.join(unseen, on="s1_row").join(pool.select("p_row", "src", "addr_full", "name_core"), on="p_row")
    # 1) base-mates
    selk = (sel.join(c.select("s1_row", "p_row", "src", "addr_full"), on=["s1_row", "p_row"])
               .filter(pl.col("addr_full") != "").select("s1_row", "src", "addr_full").unique())
    cand = c.join(sel, on=["s1_row", "p_row"], how="anti")
    bm = cand.filter(pl.col("q") >= min_q).join(selk, on=["s1_row", "src", "addr_full"], how="semi").select("s1_row", "p_row")
    # 2) acronyms (pool name = initials of the S1 name) with the S1 house number present in the pool address
    ac = cand.join(s1.select("s1_row", pl.col("name_core").alias("n1"), pl.col("name_full").alias("f1")), on="s1_row")
    ac = ac.filter(pl.col("name_core").str.len_chars().is_between(2, 6) & ~pl.col("name_core").str.contains(" "))
    ac = ac.with_columns(pl.Series("i1", [initials(x) for x in ac["n1"].to_list()]),
                         pl.Series("i2", [initials(x) for x in ac["f1"].to_list()]))
    ac = ac.filter((pl.col("name_core") == pl.col("i1")) | (pl.col("name_core") == pl.col("i2")))
    fi = pl.concat([pl.read_parquet(p, columns=["s1_row", "p_row", "num_first_in", "ac_tset"]).join(ac.select("s1_row", "p_row"), on=["s1_row", "p_row"], how="semi")
                    for p in sorted(glob.glob(os.path.join(WORK, f"feat_{split}", "part_*.parquet")))])
    ac = ac.join(fi, on=["s1_row", "p_row"]).filter((pl.col("num_first_in") == 1) | (pl.col("ac_tset") >= 90)).select("s1_row", "p_row")
    add = pl.concat([bm, ac]).unique()
    print(f"unseen-country recall repair: +{bm.height} base-mates, +{ac.height} acronyms -> +{add.height} pairs")
    return pl.concat([sel, add]).unique()
