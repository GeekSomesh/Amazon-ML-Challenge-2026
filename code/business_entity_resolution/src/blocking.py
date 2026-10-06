"""Candidate generation (blocking).

Per country (country is an open set of labels; every label present is processed):
  * each record -> sparse IDF-weighted binary vector over
      name-core word tokens ("n:" prefix)  +  address word tokens incl. numbers ("a:")
    tokens seen only once (cannot match anything) or extremely frequent are dropped.
  * top-k cosine retrieval in BOTH directions with sparse_dot_topn:
      S1 -> pool  (k_fwd per S1 entity)
      pool -> S1  (k_rev per pool record; guarantees each S2/S3 record meets its best S1s,
                   so common names cannot crowd a true match out of an S1's list)
  * union of both directions = candidate pairs, with the cosine score kept as a feature.
"""
import gc
import os
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from config import N_THREADS, WORK, gt_path, norm_path


COLS = ["entity_id", "country", "name_core", "name_join", "addr_full"]   # all that blocking reads


def load_norm(split):
    s1 = pl.read_parquet(norm_path(split, 1), columns=COLS)
    pool = pl.concat([pl.read_parquet(norm_path(split, 2), columns=COLS), pl.read_parquet(norm_path(split, 3), columns=COLS)])
    return s1, pool


def _bigrams(col):
    """overlapping adjacent token pairs of a space-separated string"""
    e = pl.col(col)
    return pl.concat_list([e.str.extract_all(r"\S+ \S+"),
                           e.str.replace(r"^\S+ ?", "").str.extract_all(r"\S+ \S+")])


def _token_exprs(bigrams, char_w):
    parts = [
        pl.col("name_core").str.split(" ").list.eval(pl.lit("n:") + pl.element()),
        pl.col("addr_full").str.split(" ").list.eval(pl.lit("a:") + pl.element()),
    ]
    if bigrams:
        parts += [_bigrams("name_core").list.eval(pl.lit("N:") + pl.element()),
                  _bigrams("addr_full").list.eval(pl.lit("A:") + pl.element())]
    return parts


def _tokens(df, bigrams=True, char_w=0.0, combo_w=0.0, chunk=1_000_000):
    """(row, token-id) pairs; token strings are turned into categorical ids chunk by chunk
    (under a global StringCache) so that peak memory stays small."""
    out = []
    for off in range(0, df.height, chunk):
        sub = df.slice(off, chunk)
        t = (sub.select(pl.int_range(off, off + sub.height, dtype=pl.UInt32).alias("row"),
                        pl.concat_list(_token_exprs(bigrams, char_w)).alias("tok"))
                .explode("tok").filter(pl.col("tok").is_not_null() & (pl.col("tok").str.len_chars() > 2)))
        if char_w > 0:
            c = (sub.select(pl.int_range(off, off + sub.height, dtype=pl.UInt32).alias("row"), pl.col("name_join"))
                    .with_columns(pl.int_ranges(0, (pl.col("name_join").str.len_chars().cast(pl.Int64) - 3).clip(0, None)).alias("i"))
                    .explode("i").drop_nulls("i")
                    .select("row", (pl.lit("c:") + pl.col("name_join").str.slice(pl.col("i"), 4)).alias("tok")))
            t = pl.concat([t, c])
        if combo_w > 0:
            # composite name-token x address-number keys ("vec|204"): recover records whose
            # address is truncated to a number + city while the name is very common
            rid = pl.int_range(off, off + sub.height, dtype=pl.UInt32).alias("row")
            nt = (sub.select(rid, pl.col("name_core").str.split(" ").alias("nt")).explode("nt")
                     .filter(pl.col("nt").str.len_chars() >= 2))
            nm = sub.select(rid, pl.col("addr_full").str.extract_all(r"\b\d+\b").alias("num")).explode("num").drop_nulls("num")
            k = nt.join(nm, on="row").select("row", (pl.lit("K:") + pl.col("nt") + pl.lit("|") + pl.col("num")).alias("tok"))
            t = pl.concat([t, k])
        t = t.with_columns(pl.col("tok").cast(pl.Categorical)).unique(["row", "tok"])
        out.append((off, sub.height, t))
    return out


def build_vectors(s1c, poolc, w_name=1.0, w_addr=1.0, max_df=5000, min_df=2, bigrams=True, char_w=0.0, combo_w=0.0):
    with pl.StringCache():
        t1 = _tokens(s1c, bigrams, char_w, combo_w)
        t2 = _tokens(poolc, bigrams, char_w, combo_w)
        n = s1c.height + poolc.height
        # document frequencies summed over chunks (peak memory stays at one chunk)
        dfc = (pl.concat([t.group_by("tok").len() for _, _, t in t1 + t2])
                 .group_by("tok").agg(pl.col("len").sum()))
        dfc = dfc.filter((pl.col("len") >= min_df) & (pl.col("len") <= max_df))
        ts = dfc["tok"].cast(pl.String)
        mult = np.where(ts.str.starts_with("c:").to_numpy(), char_w,
                        np.where(ts.str.starts_with("K:").to_numpy(), combo_w,
                                 np.where(ts.str.to_lowercase().str.starts_with("n:").to_numpy(), w_name, w_addr)))
        dfc = dfc.with_columns(
            pl.int_range(pl.len(), dtype=pl.Int32).alias("col"),
            ((pl.lit(n) / pl.col("len")).log() * pl.Series(mult)).cast(pl.Float32).alias("w"),
        )
        nv = dfc.height

        voc = dfc.select("tok", "col", "w")

        def csr(chunks):
            # one CSR block per token chunk, stacked: never materialises the whole COO at once
            blocks = []
            for off, h, t in chunks:
                j = t.join(voc, on="tok", how="inner")
                m = sp.csr_matrix((j["w"].to_numpy(), ((j["row"].to_numpy() - off).astype(np.int64), j["col"].to_numpy())),
                                  shape=(h, nv), dtype=np.float32)
                del j
                norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
                norms[norms == 0] = 1.0
                blocks.append(sp.diags((1.0 / norms).astype(np.float32)).dot(m).tocsr())
            return sp.vstack(blocks, format="csr")

        A1 = csr(t1)
        del t1
        A2 = csr(t2)
        del t2
        return A1, A2


def topk(A, B, k, threshold=0.0, chunk=500_000):
    """rows of A vs rows of B -> (row_idx_A, row_idx_B, score) of top-k per A row."""
    BT = B.T.tocsr()
    ra, rb, sc = [], [], []
    for off in range(0, A.shape[0], chunk):
        C = sp_matmul_topn(A[off:off + chunk], BT, top_n=k, threshold=threshold, n_threads=N_THREADS)
        C = C.tocoo()
        ra.append(C.row.astype(np.int64) + off)
        rb.append(C.col.astype(np.int64))
        sc.append(C.data.astype(np.float32))
    return np.concatenate(ra), np.concatenate(rb), np.concatenate(sc)


def generate(split, k_fwd=20, k_rev=6, threshold=0.02, prune=None, **vec_kw):
    """prune=(a, b): keep a pair only if it is in the top-a of its S1 list or top-b of its pool list."""
    t0 = time.time()
    s1, pool = load_norm(split)
    s1 = s1.with_row_index("s1_row")
    pool = pool.with_row_index("p_row")
    out = []
    for ctry in sorted(set(s1["country"].unique().to_list()) | set(pool["country"].unique().to_list())):
        s1c = s1.filter(pl.col("country") == ctry)
        pc = pool.filter(pl.col("country") == ctry)
        if s1c.height == 0 or pc.height == 0:
            continue
        t = time.time()
        A1, A2 = build_vectors(s1c, pc, **vec_kw)
        tb = time.time() - t
        t = time.time()
        f1, f2, fs = topk(A1, A2, k_fwd, threshold)          # S1 -> pool
        r2, r1, rs = topk(A2, A1, k_rev, threshold)          # pool -> S1
        tt = time.time() - t
        s1_rows = s1c["s1_row"].to_numpy()
        p_rows = pc["p_row"].to_numpy()
        c = pl.DataFrame({
            "s1_row": np.concatenate([s1_rows[f1], s1_rows[r1]]).astype(np.uint32),
            "p_row": np.concatenate([p_rows[f2], p_rows[r2]]).astype(np.uint32),
            "blk_cos": np.concatenate([fs, rs]),
        }).unique(["s1_row", "p_row"])
        if prune is not None:
            c = c.filter((pl.col("blk_cos").rank("ordinal", descending=True).over("s1_row") <= prune[0])
                         | (pl.col("blk_cos").rank("ordinal", descending=True).over("p_row") <= prune[1]))
        out.append(c)
        print(f"  [{split}] {ctry}: S1={s1c.height} pool={pc.height} vocab={A1.shape[1]} "
              f"nnz/row={A1.nnz / max(1, A1.shape[0]):.1f}/{A2.nnz / max(1, A2.shape[0]):.1f} "
              f"build={tb:.0f}s topk={tt:.0f}s cands={c.height} ({c.height / s1c.height:.1f}/S1)", flush=True)
        del A1, A2, f1, f2, fs, r1, r2, rs
        gc.collect()
    cand = pl.concat(out)
    print(f"[{split}] total candidates {cand.height} ({cand.height / s1.height:.2f}/S1) in {time.time() - t0:.0f}s", flush=True)
    return s1.select("s1_row", "entity_id", "country"), pool.select("p_row", "entity_id", "country"), cand


def gt_pairs(s1, pool, split="train"):
    gt = pl.read_parquet(gt_path(split))
    ex = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
            .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "p_id"}))
    ex = (ex.join(s1.select(pl.col("entity_id").alias("s1_id"), "s1_row"), on="s1_id")
            .join(pool.select(pl.col("entity_id").alias("p_id"), "p_row"), on="p_id"))
    return ex.select("s1_row", "p_row")


def recall_report(cand, gt, s1, label=""):
    hit = gt.join(cand.select("s1_row", "p_row", "blk_cos"), on=["s1_row", "p_row"], how="left")
    r = hit["blk_cos"].is_not_null().mean()
    by_c = (hit.join(s1.select("s1_row", "country"), on="s1_row")
               .group_by("country").agg(pl.col("blk_cos").is_not_null().mean().alias("recall"), pl.len()))
    print(f"{label} pair recall={r:.5f}  cands/S1={cand.height / s1.height:.2f}  by country: "
          f"{dict(zip(by_c['country'].to_list(), [round(x, 5) for x in by_c['recall'].to_list()]))}", flush=True)
    return r


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--k_fwd", type=int, default=20)
    ap.add_argument("--k_rev", type=int, default=6)
    ap.add_argument("--max_df", type=int, default=3000)
    ap.add_argument("--char_w", type=float, default=0.0)
    ap.add_argument("--combo_w", type=float, default=0.0)
    ap.add_argument("--prune_a", type=int, default=0)
    ap.add_argument("--prune_b", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    prune = (args.prune_a, args.prune_b) if args.prune_a > 0 else None
    s1, pool, cand = generate(args.split, k_fwd=args.k_fwd, k_rev=args.k_rev, prune=prune,
                              max_df=args.max_df, char_w=args.char_w, combo_w=args.combo_w)
    cand.write_parquet(os.path.join(WORK, args.out or f"{args.split}_cand.parquet"))
    if args.split in ("train", "trainsim"):
        recall_report(cand, gt_pairs(s1, pool, args.split), s1, args.split)
