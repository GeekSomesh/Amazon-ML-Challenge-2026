"""LaBSE sibling similarity: for each stage-2 pair (S1, p), the name similarity between p and the S1's other
confident candidates (top-3 by stage-1 probability, prob >= 0.3). Copies of one business share a per-source base
record, so a genuine copy - even with a blank address - resembles its siblings; a same-name record of another
business resembles them less.

  python emb_sib.py trainsim
  python emb_sib.py test <stage1_tag>
-> work/emb_sib_<split>.parquet: emb_sib_max, emb_sib_same_src, emb_sib_w (max cos x sibling prob)
"""
import os
import sys

import numpy as np
import polars as pl

from config import WORK, norm_path
from stage2 import MIN_P1
from train_stage2 import cached_frame

TOPQ = 3


def main(split, stage1_tag=None):
    names = pl.read_parquet(os.path.join(WORK, "emb_names.parquet")).select(pl.col("eid").cast(pl.UInt32), "name_full")
    E = np.asarray(np.load(os.path.join(WORK, "emb_names_128.npy"), mmap_mode="r"))
    if split == "trainsim":
        pr = pl.read_parquet(cached_frame(os.environ.get("BER_STAGE1", "e12")), columns=["s1_row", "p_row", "prob1"])
    else:
        pr = (pl.read_parquet(os.path.join(WORK, f"test_scored_{stage1_tag}.parquet"), columns=["s1_row", "p_row", "prob"])
                .filter(pl.col("prob") >= MIN_P1).rename({"prob": "prob1"}))
    pool = pl.concat([pl.read_parquet(norm_path(split, s), columns=["name_full"]) for s in (2, 3)]).with_row_index("p_row").with_columns(
        pl.col("p_row").cast(pl.UInt32), pl.col("name_full").fill_null(""))
    n2 = pl.scan_parquet(norm_path(split, 2)).select(pl.len()).collect().item()
    pe = (pool.join(pr.select("p_row").unique(), on="p_row", how="semi").join(names, on="name_full", how="left")
              .select("p_row", "eid", (pl.col("p_row") < n2).alias("is_s2")))
    missing = pe["eid"].null_count()
    if missing:
        print(f"warning: {missing} pool names without embedding (treated as no-sibling)", flush=True)
    pr = pr.join(pe, on="p_row", how="left")
    top = (pr.filter(pl.col("prob1") >= 0.3).sort(["s1_row", "prob1"], descending=[False, True])
             .group_by("s1_row", maintain_order=True).head(TOPQ + 1)
             .select("s1_row", pl.col("p_row").alias("q_row"), pl.col("eid").alias("q_eid"), pl.col("is_s2").alias("q_s2"), pl.col("prob1").alias("q_prob")))
    x = pr.select("s1_row", "p_row", "eid", "is_s2").join(top, on="s1_row").filter((pl.col("p_row") != pl.col("q_row")) & pl.col("eid").is_not_null() & pl.col("q_eid").is_not_null())
    a, b = x["eid"].to_numpy(), x["q_eid"].to_numpy()
    cos = np.empty(x.height, dtype=np.float32)
    for off in range(0, x.height, 1_000_000):
        cos[off:off + 1_000_000] = (E[a[off:off + 1_000_000]].astype(np.float32) * E[b[off:off + 1_000_000]].astype(np.float32)).sum(axis=1)
    x = x.with_columns(pl.Series("c", cos))
    agg = x.group_by("s1_row", "p_row").agg(
        pl.col("c").max().alias("emb_sib_max"),
        pl.col("c").filter(pl.col("is_s2") == pl.col("q_s2")).max().alias("emb_sib_same_src"),
        (pl.col("c") * pl.col("q_prob")).max().alias("emb_sib_w"))
    out = os.path.join(WORK, f"emb_sib_{split}.parquet")
    agg.write_parquet(out)
    print(f"wrote {out}: {agg.height} pairs with siblings (of {pr.height})", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
