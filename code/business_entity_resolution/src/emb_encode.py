"""Multilingual sentence embeddings (LaBSE, Apache-2.0) of every normalised name / address that occurs in a
stage-2 pair (trainsim frame + test candidates): language-agnostic similarity that transfers from the training
countries (US, India) to unseen ones (France).

  python emb_encode.py              -> work/emb_names.parquet (string, eid) + work/emb_names_128.npy
  python emb_encode.py addr_full    -> work/emb_addr.parquet  (string, eid) + work/emb_addr_128.npy
Vectors: L2-normalised LaBSE output projected to 128 dims by PCA, stored as float16. Requires a CUDA GPU.
"""
import os
import sys
import time

import numpy as np
import polars as pl

from config import WORK, norm_path
from stage2 import MIN_P1
from train_stage2 import cached_frame

DIM = 128
CHUNK = 200_000
COL = sys.argv[1] if len(sys.argv) > 1 else "name_full"
PRE = "emb_names" if COL == "name_full" else "emb_addr"


def pairs_names(stage1_tag=os.environ.get("BER_STAGE1", "e12")):
    """every normalised string (COL) of a stage-2 pair: trainsim frame + test pairs with stage-1 prob >= MIN_P1"""
    fr = pl.read_parquet(cached_frame(stage1_tag), columns=["s1_row", "p_row"])
    te = (pl.read_parquet(os.path.join(WORK, f"test_scored_{stage1_tag}.parquet"), columns=["s1_row", "p_row", "prob"])
            .filter(pl.col("prob") >= MIN_P1).select("s1_row", "p_row"))
    out = []
    for split, pr in (("trainsim", fr), ("test", te)):
        s1 = pl.read_parquet(norm_path(split, 1), columns=[COL]).with_row_index("r")
        pool = pl.concat([pl.read_parquet(norm_path(split, s), columns=[COL]) for s in (2, 3)]).with_row_index("r")
        out.append(s1.join(pr.select(pl.col("s1_row").cast(pl.UInt32).alias("r")).unique(), on="r", how="semi")[COL])
        out.append(pool.join(pr.select(pl.col("p_row").cast(pl.UInt32).alias("r")).unique(), on="r", how="semi")[COL])
    u = pl.concat(out).fill_null("").unique().sort()
    return pl.DataFrame({"name_full": u}).with_row_index("eid")   # column named name_full for both kinds


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    names = pairs_names()
    names.write_parquet(os.path.join(WORK, f"{PRE}.parquet"))
    n = names.height
    print(f"{n} unique names ({time.time() - t0:.0f}s)", flush=True)
    m = SentenceTransformer("sentence-transformers/LaBSE", device="cuda")
    m.half()
    m.max_seq_length = 32 if COL == "name_full" else 48
    out = np.lib.format.open_memmap(os.path.join(WORK, f"{PRE}_128.npy"), mode="w+", dtype=np.float16, shape=(n, DIM))
    txt = names["name_full"].to_list()
    proj = None
    if COL != "name_full":   # addresses: fit the PCA on a random sample (sorted chunks are not representative)
        rs = np.random.default_rng(0).choice(n, size=min(120_000, n), replace=False)
        e = m.encode([txt[i] for i in rs], batch_size=512, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        mu = e.mean(axis=0)
        _, _, vt = np.linalg.svd(e - mu, full_matrices=False)
        proj = vt[:DIM].T.astype(np.float32)
        np.save(os.path.join(WORK, "emb_addr_pca.npy"), np.vstack([mu[None, :], proj.T]))
    for off in range(0, n, CHUNK):
        e = m.encode(txt[off:off + CHUNK], batch_size=512, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        if proj is None:   # PCA on the first chunk (names are sorted -> use a strided sample of it)
            mu = e.mean(axis=0)
            _, _, vt = np.linalg.svd(e[::2] - mu, full_matrices=False)
            proj = vt[:DIM].T.astype(np.float32)
            np.save(os.path.join(WORK, "emb_pca.npy" if COL == "name_full" else "emb_addr_pca.npy"), np.vstack([mu[None, :], proj.T]))
        z = (e - mu) @ proj
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        out[off:off + len(z)] = z.astype(np.float16)
        if (off // CHUNK) % 5 == 0:
            done = off + len(z)
            print(f"  {done}/{n} ({time.time() - t0:.0f}s, eta {(time.time() - t0) / done * (n - done) / 60:.0f} min)", flush=True)
        torch.cuda.empty_cache()
    out.flush()
    print(f"EMB_DONE {n} names in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
