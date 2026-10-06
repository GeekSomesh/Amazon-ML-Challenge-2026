"""Convert the raw TSV files (Source 1/2/3 for train and test, training ground truth) to parquet."""
import os
import sys
import time

import polars as pl

from config import DATA, WORK


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       has_header=True, encoding="utf8", missing_utf8_is_empty_string=True)


def main():
    os.makedirs(WORK, exist_ok=True)
    for split in ("train", "test"):
        for src in (1, 2, 3):
            t = time.time()
            p = os.path.join(DATA, split, f"{split}_source{src}.tsv")
            df = read_tsv(p)
            df.write_parquet(os.path.join(WORK, f"{split}_s{src}.parquet"))
            print(split, src, df.shape, f"{time.time()-t:.1f}s", flush=True)
    t = time.time()
    gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))
    gt.write_parquet(os.path.join(WORK, "train_gt.parquet"))
    print("gt", gt.shape, f"{time.time()-t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
