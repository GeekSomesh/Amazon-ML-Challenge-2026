"""Pair features from the LaBSE name embeddings (see emb_encode.py).

  python emb_pairs.py trainsim            -> work/emb_feat_trainsim.parquet   (pairs of the cached stage-2 frame)
  python emb_pairs.py test <stage1_tag>   -> work/emb_feat_test.parquet       (pairs with stage-1 prob >= MIN_P1)
features: emb_cos (S1 name vs pool name), emb_rank_s1 (rank among the S1's candidates), emb_gap_s1 (cos minus the
best OTHER candidate of the S1), emb_gap_p (cos minus the best OTHER S1 of the pool record)
"""
import os
import sys

import numpy as np
import polars as pl

from config import WORK, norm_path
from stage2 import MIN_P1
from train_stage2 import cached_frame

KIND = os.environ.get("EMB_KIND", "name")
COL = "name_full" if KIND == "name" else "addr_full"
PRE = "emb_names" if KIND == "name" else "emb_addr"
FP = "emb" if KIND == "name" else "emba"


def embed_missing(names: pl.DataFrame, need: pl.Series, E: np.ndarray):
    """encode strings not yet embedded (e.g. after a normalisation change) and append them"""
    miss = need.unique().to_frame("name_full").join(names, on="name_full", how="anti")
    if miss.height == 0:
        return names, E
    from sentence_transformers import SentenceTransformer
    pca = np.load(os.path.join(WORK, "emb_pca.npy" if KIND == "name" else "emb_addr_pca.npy"))
    mu, proj = pca[0], pca[1:].T
    m = SentenceTransformer("sentence-transformers/LaBSE", device="cuda")
    m.half()
    m.max_seq_length = 32 if KIND == "name" else 48
    e = m.encode(miss["name_full"].to_list(), batch_size=512, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    z = (e - mu) @ proj
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    base = names.height
    names = pl.concat([names, miss.with_row_index("eid", offset=base).select("eid", "name_full")])
    print(f"embedded {miss.height} new names", flush=True)
    return names, np.vstack([E, z.astype(np.float16)])


def main(split, stage1_tag=None):
    names = pl.read_parquet(os.path.join(WORK, f"{PRE}.parquet")).select(pl.col("eid").cast(pl.UInt32), "name_full")
    E = np.load(os.path.join(WORK, f"{PRE}_128.npy"), mmap_mode="r")
    E = np.asarray(E)
    if split == "trainsim":
        pr = pl.read_parquet(cached_frame(os.environ.get("BER_STAGE1", "e12")), columns=["s1_row", "p_row", "prob1"]).rename({"prob1": "prob"})
    else:
        pr = pl.read_parquet(os.path.join(WORK, f"test_scored_{stage1_tag}.parquet"), columns=["s1_row", "p_row", "prob"]).filter(pl.col("prob") >= MIN_P1)
    s1 = pl.read_parquet(norm_path(split, 1), columns=[COL]).with_row_index("s1_row").rename({COL: "name_full"}).with_columns(pl.col("s1_row").cast(pl.UInt32), pl.col("name_full").fill_null(""))
    pool = pl.concat([pl.read_parquet(norm_path(split, s), columns=[COL]) for s in (2, 3)]).rename({COL: "name_full"}).with_row_index("p_row").with_columns(
        pl.col("p_row").cast(pl.UInt32), pl.col("name_full").fill_null(""))
    d = (pr.join(s1, on="s1_row").rename({"name_full": "n1"}).join(pool, on="p_row").rename({"name_full": "n2"}))
    names, E = embed_missing(names, pl.concat([d["n1"], d["n2"]]), E)
    d = (d.join(names.rename({"name_full": "n1", "eid": "e1"}), on="n1", how="left")
          .join(names.rename({"name_full": "n2", "eid": "e2"}), on="n2", how="left"))
    e1, e2 = d["e1"].to_numpy(), d["e2"].to_numpy()
    cos = np.empty(d.height, dtype=np.float32)
    for off in range(0, d.height, 1_000_000):
        a = E[e1[off:off + 1_000_000]].astype(np.float32)
        b = E[e2[off:off + 1_000_000]].astype(np.float32)
        cos[off:off + 1_000_000] = (a * b).sum(axis=1)
    d = d.select("s1_row", "p_row").with_columns(pl.Series(f"{FP}_cos", cos))
    c = pl.col(f"{FP}_cos")
    d = d.with_columns(c.rank("ordinal", descending=True).over("s1_row").cast(pl.Float32).alias(f"{FP}_rank_s1"),
                       c.top_k(2).over("s1_row", mapping_strategy="join").alias("_t2s"),
                       c.top_k(2).over("p_row", mapping_strategy="join").alias("_t2p"))
    other = lambda t: pl.when(c >= pl.col(t).list.get(0)).then(pl.col(t).list.get(1, null_on_oob=True)).otherwise(pl.col(t).list.get(0))
    d = d.with_columns((c - other("_t2s").fill_null(-1.0)).alias(f"{FP}_gap_s1"), (c - other("_t2p").fill_null(-1.0)).alias(f"{FP}_gap_p")).drop("_t2s", "_t2p")
    out = os.path.join(WORK, f"{FP}_feat_{split}.parquet")
    d.write_parquet(out)
    print(f"wrote {out}: {d.height} pairs; cos mean {d[FP + '_cos'].mean():.3f}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
