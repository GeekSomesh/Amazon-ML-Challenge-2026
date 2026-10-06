"""Embed and PERSIST every normalised name of a split's stage-2 pairs that is not yet in emb_names (e.g. after a
normalisation change), so that all LaBSE features (pair and sibling) see the same embeddings.
  python emb_complete.py test <stage1_tag>
"""
import os
import sys

import numpy as np
import polars as pl

from config import WORK, norm_path
from emb_pairs import embed_missing
from stage2 import MIN_P1

KIND = os.environ.get("EMB_KIND", "name")
COL = "name_full" if KIND == "name" else "addr_full"
PRE = "emb_names" if KIND == "name" else "emb_addr"
FP = "emb" if KIND == "name" else "emba"


def main(split, stage1_tag):
    names = pl.read_parquet(os.path.join(WORK, f"{PRE}.parquet")).select(pl.col("eid").cast(pl.UInt32), "name_full")
    E = np.asarray(np.load(os.path.join(WORK, f"{PRE}_128.npy"), mmap_mode="r"))
    pr = pl.read_parquet(os.path.join(WORK, f"test_scored_{stage1_tag}.parquet"), columns=["s1_row", "p_row", "prob"]).filter(pl.col("prob") >= MIN_P1)
    s1 = pl.read_parquet(norm_path(split, 1), columns=[COL]).with_row_index("s1_row").rename({COL: "name_full"}).with_columns(pl.col("s1_row").cast(pl.UInt32), pl.col("name_full").fill_null(""))
    pool = pl.concat([pl.read_parquet(norm_path(split, s), columns=[COL]) for s in (2, 3)]).rename({COL: "name_full"}).with_row_index("p_row").with_columns(
        pl.col("p_row").cast(pl.UInt32), pl.col("name_full").fill_null(""))
    need = pl.concat([s1.join(pr.select("s1_row").unique(), on="s1_row", how="semi")["name_full"],
                      pool.join(pr.select("p_row").unique(), on="p_row", how="semi")["name_full"]])
    n0 = names.height
    names, E = embed_missing(names, need, E)
    if names.height == n0:
        print("nothing missing", flush=True)
        return
    del_path = os.path.join(WORK, f"{PRE}_128.npy")
    np.save(del_path + ".tmp.npy", E)
    names.write_parquet(os.path.join(WORK, f"{PRE}.parquet"))
    os.replace(del_path + ".tmp.npy", del_path)
    print(f"persisted {names.height - n0} new names (total {names.height})", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
