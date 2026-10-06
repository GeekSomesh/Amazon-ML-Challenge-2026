"""Compute pairwise + context features for every candidate pair of a split.

Phase A: candidate table -> context features (+ labels / folds for training splits),
         sorted by S1 row and written to disk, then freed.
Phase B: record store (normalised strings, token CSR, IDF) -> pair features chunk by chunk.
Output: work/feat_{split}/part_XXX.parquet with columns
  s1_row, p_row, blk_cos, <context feats>, [label, fold], <pair feats>
"""
import gc
import os
import shutil
import sys
import time

import numpy as np
import polars as pl

from config import WORK, gt_path, norm_path
from features import Store, context_features, pair_features

CHUNK = 2_000_000


def fold_of(ids: pl.Series, k=2):
    return (ids.hash(seed=7) % k).cast(pl.Int8)


def gt_rows(split):
    s1 = pl.read_parquet(norm_path(split, 1), columns=["entity_id"]).with_row_index("s1_row")
    pool = pl.concat([pl.read_parquet(norm_path(split, 2), columns=["entity_id"]),
                      pl.read_parquet(norm_path(split, 3), columns=["entity_id"])]).with_row_index("p_row")
    gt = pl.read_parquet(gt_path(split))
    ex = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
            .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "p_id"}))
    ex = (ex.join(s1.rename({"entity_id": "s1_id"}), on="s1_id").join(pool.rename({"entity_id": "p_id"}), on="p_id"))
    folds = s1.select(pl.col("s1_row").cast(pl.UInt32), fold_of(s1["entity_id"]).alias("fold"))
    return (ex.select(pl.col("s1_row").cast(pl.UInt32), pl.col("p_row").cast(pl.UInt32), pl.lit(1, pl.Int8).alias("label")),
            folds)


def phase_a(split, cand_file):
    t = time.time()
    cand = pl.read_parquet(os.path.join(WORK, cand_file))
    cand = context_features(cand, "blk_cos", "blk")
    if split != "test":
        lab, folds = gt_rows(split)
        cand = cand.join(lab, on=["s1_row", "p_row"], how="left").with_columns(pl.col("label").fill_null(0))
        cand = cand.join(folds, on="s1_row", how="left")
        print(f"labels: {int(cand['label'].sum())} positives / {cand.height} pairs "
              f"(GT pairs {lab.height}, blocking recall {cand['label'].sum() / lab.height:.5f})", flush=True)
    cand = cand.sort(["s1_row", "p_row"])
    path = os.path.join(WORK, f"{split}_cand_ctx.parquet")
    cand.write_parquet(path)
    n = cand.height
    del cand
    gc.collect()
    print(f"phase A done in {time.time() - t:.0f}s ({n} pairs)", flush=True)
    return path, n


def phase_b(split, path, n):
    t0 = time.time()
    store = Store(split)
    out_dir = os.path.join(WORK, f"feat_{split}")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    lf = pl.scan_parquet(path)
    for i, off in enumerate(range(0, n, CHUNK)):
        t = time.time()
        c = lf.slice(off, CHUNK).collect()
        f = pair_features(store, c["s1_row"].to_numpy(), c["p_row"].to_numpy())
        pl.concat([c, f], how="horizontal").write_parquet(os.path.join(out_dir, f"part_{i:03d}.parquet"), compression="zstd")
        print(f"  part {i}: {c.height} pairs in {time.time() - t:.0f}s", flush=True)
    print(f"[{split}] features done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    split, cand_file = sys.argv[1], sys.argv[2]
    path, n = phase_a(split, cand_file)
    phase_b(split, path, n)
