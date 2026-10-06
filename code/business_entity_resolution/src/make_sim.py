"""Create the 'trainsim' split: training data re-shaped to look like the test set.

EDA: pool records per Source-1 entity are 4.68 in train but 5.75 in test, i.e. the
test set carries ~2x more distractors (Source-2/3 records whose entity is not in
Source 1). We reproduce that by removing a random 18.7% of Source-1 entities BEFORE
blocking; their Source-2/3 records stay in the pool as realistic orphan distractors
(same names at other addresses, co-located businesses, etc.). All model training,
validation and threshold tuning happen on this split.
"""
import os

import polars as pl

from config import WORK

KEEP_FRAC = float(os.environ.get("BER_KEEP_FRAC", str(4.6765 / 5.7543)))  # 1.0 = keep all S1 (test-like: extra records are planted decoys)


def main(seed=2026):
    s1 = pl.read_parquet(os.path.join(WORK, "train_s1_norm.parquet"))
    keep = s1.sample(fraction=KEEP_FRAC, seed=seed, shuffle=False)["entity_id"]
    s1k = s1.filter(pl.col("entity_id").is_in(keep.implode()))
    s1k.write_parquet(os.path.join(WORK, "trainsim_s1_norm.parquet"))
    gt = pl.read_parquet(os.path.join(WORK, "train_gt.parquet"))
    gt = gt.filter(pl.col("source1_entity_id").is_in(keep.implode()))
    gt.write_parquet(os.path.join(WORK, "trainsim_gt.parquet"))
    n_pool = sum(pl.scan_parquet(os.path.join(WORK, f"train_s{i}_norm.parquet")).select(pl.len()).collect().item() for i in (2, 3))
    print(f"trainsim: kept {s1k.height}/{s1.height} S1 entities; pool/S1 = {n_pool / s1k.height:.3f}")


if __name__ == "__main__":
    main()
