"""Pairwise features for (Source-1 record, Source-2/3 record) candidate pairs.

Fast paths:
  * string similarities: rapidfuzz.process.cpdist (C++, multi-threaded, element-wise)
  * IDF-weighted token overlap, number agreement: numba kernels over CSR token-id arrays
  * context / competition features: polars window aggregations over the candidate table
Every feature is language/country agnostic (IDF is computed per country from the data
itself), so the model transfers to country labels never seen in training (France).
"""
import os
import time

import numba as nb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from config import WORK, norm_path
from postfilter import MARKERS_TRAIN


# ============================================================================ store
class Store:
    """Normalised records of one split + token-id CSR arrays + per-country IDF."""

    def __init__(self, split):
        t = time.time()
        s1 = pl.read_parquet(norm_path(split, 1))
        pool = pl.concat([pl.read_parquet(norm_path(split, 2)), pl.read_parquet(norm_path(split, 3))])
        self.n1, self.np_ = s1.height, pool.height
        self.s1 = s1
        self.pool = pool
        self.s1_id = s1["entity_id"]
        self.p_id = pool["entity_id"]
        both = pl.concat([s1.select("country", "name_core", "name_sorted", "addr_full"),
                          pool.select("country", "name_core", "name_sorted", "addr_full")])
        # token CSR (country-scoped vocab so IDF is per country)
        self.name_ptr, self.name_ids, self.name_idf = self._csr(both, "name_core")
        self.addr_ptr, self.addr_ids, self.addr_idf = self._csr(both, "addr_full")
        self.num_ptr, self.num_vals, self.num_first = self._nums(both["addr_full"])
        # "address core": address without each country's very frequent tokens (state / region /
        # department names, street-type words). Generic, so it transfers to unseen countries
        # (e.g. France pool records naming the department where Source 1 names the region).
        freq = self._voc["addr_full"].filter(pl.col("frac") > 0.02)["tok"].unique().to_list()
        core = lambda df: df.with_columns(pl.col("addr_full").str.split(" ")
                                          .list.eval(pl.element().filter(~pl.element().is_in(freq))).list.join(" ").alias("addr_core"))
        self.s1 = core(self.s1)
        self.pool = core(self.pool)
        self.addr_freq_tokens = freq
        # frequency / commonness features (per country, counted in S1 and in pool)
        fr = both.with_columns(pl.concat([pl.zeros(self.n1, dtype=pl.Int8, eager=True),
                                          pl.ones(self.np_, dtype=pl.Int8, eager=True)]).alias("is_pool"))
        fr = fr.with_columns(
            (pl.len().over("country", "name_sorted") - pl.col("is_pool").sum().over("country", "name_sorted")).alias("name_f1"),
            pl.col("is_pool").sum().over("country", "name_sorted").alias("name_fp"),
            (pl.len().over("country", "addr_full") - pl.col("is_pool").sum().over("country", "addr_full")).alias("addr_f1"),
            pl.col("is_pool").sum().over("country", "addr_full").alias("addr_fp"),
        )
        self.name_f1 = fr["name_f1"].to_numpy().astype(np.float32)
        self.name_fp = fr["name_fp"].to_numpy().astype(np.float32)
        self.addr_f1 = fr["addr_f1"].to_numpy().astype(np.float32)
        self.addr_fp = fr["addr_fp"].to_numpy().astype(np.float32)
        # generated / random names: share of pool name tokens (len>=4) never used in any Source-1 name
        # of the same country (random replacement names like "evokor", glued domains)
        voc = (s1.select("country", pl.col("name_core").str.split(" ").alias("t")).explode("t")
                 .filter(pl.col("t").str.len_chars() >= 4).unique().with_columns(pl.lit(1).alias("_seen")))
        pt = (pool.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("r"), "country", pl.col("name_core").str.split(" ").alias("t"))
                  .explode("t").filter(pl.col("t").str.len_chars() >= 4)
                  .join(voc, on=["country", "t"], how="left").group_by("r").agg((1 - pl.col("_seen").fill_null(0)).mean().alias("u")))
        un = np.full(self.np_, -1.0, dtype=np.float32)
        un[pt["r"].to_numpy()] = pt["u"].to_numpy()
        self.p_unseen = un
        # acronym support: initials of the Source-1 name (with and without legal words), and the
        # Source-1 full name glued together (domains that keep the legal form, e.g. 444compagniesarl.com)
        stop = {"and", "et", "of", "de", "du", "des", "la", "le", "les", "the", "l", "d"}
        ini = lambda s: "".join(w[0] for w in s.split() if w and w not in stop)
        self.s1 = self.s1.with_columns(pl.Series("init_core", [ini(x) for x in s1["name_core"].to_list()]),
                                       pl.Series("init_full", [ini(x) for x in s1["name_full"].to_list()]),
                                       pl.col("name_full").str.replace_all(" ", "").alias("full_join"))
        print(f"Store({split}) built in {time.time() - t:.0f}s", flush=True)

    def _csr(self, both, col, chunk=2_000_000):
        """Token-id CSR with a country-scoped vocabulary (IDF per country).
        Pass 1 counts document frequency per (country, token) chunk by chunk; pass 2 maps
        every chunk to the shared integer ids (keeps peak memory low)."""
        def exploded(sub, off):
            return (sub.select(pl.int_range(off, off + sub.height, dtype=pl.UInt32).alias("row"), "country",
                               pl.col(col).str.split(" ").alias("tok"))
                       .explode("tok").filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
                       .unique(["row", "tok"]))
        counts = [exploded(both.slice(off, chunk), off).group_by("country", "tok").len()
                  for off in range(0, both.height, chunk)]
        voc = pl.concat(counts).group_by("country", "tok").agg(pl.col("len").sum())
        ncty = both.group_by("country").len().rename({"len": "N"})
        voc = (voc.join(ncty, on="country")
                  .with_columns(pl.int_range(pl.len(), dtype=pl.Int32).alias("tid"),
                                (pl.col("N") / pl.col("len")).log().cast(pl.Float32).alias("idf"),
                                (pl.col("len") / pl.col("N")).alias("frac")))
        parts = [exploded(both.slice(off, chunk), off).join(voc.select("country", "tok", "tid"), on=["country", "tok"])
                                                     .select("row", "tid")
                 for off in range(0, both.height, chunk)]
        t = pl.concat(parts).sort(["row", "tid"])
        del parts
        rows = t["row"].to_numpy()
        ptr = np.zeros(both.height + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows, minlength=both.height), out=ptr[1:])
        idf = np.zeros(voc.height, dtype=np.float32)
        idf[voc["tid"].to_numpy()] = voc["idf"].to_numpy()
        self._voc = getattr(self, "_voc", {})
        self._voc[col] = voc.select("country", "tok", "frac")
        return ptr, t["tid"].to_numpy().astype(np.int32), idf

    def _nums(self, addr):
        nums = addr.str.extract_all(r"\b\d+\b")
        first = nums.list.first().cast(pl.Int64, strict=False).fill_null(-1).to_numpy()
        t = (pl.DataFrame({"row": np.arange(len(addr), dtype=np.uint32), "n": nums})
               .explode("n").filter(pl.col("n").is_not_null())
               .with_columns(pl.col("n").str.slice(0, 15).cast(pl.Int64)).unique(["row", "n"]).sort(["row", "n"]))
        ptr = np.zeros(len(addr) + 1, dtype=np.int64)
        np.cumsum(np.bincount(t["row"].to_numpy(), minlength=len(addr)), out=ptr[1:])
        return ptr, t["n"].to_numpy(), first


# ============================================================================ numba kernels
@nb.njit(parallel=True, cache=True)
def _overlap(ptr, ids, idf, ra, rb):
    n = ra.shape[0]
    out = np.zeros((n, 6), dtype=np.float32)
    for i in nb.prange(n):
        a0, a1 = ptr[ra[i]], ptr[ra[i] + 1]
        b0, b1 = ptr[rb[i]], ptr[rb[i] + 1]
        wa = 0.0
        wb = 0.0
        wi = 0.0
        ci = 0
        for k in range(a0, a1):
            wa += idf[ids[k]]
        for k in range(b0, b1):
            wb += idf[ids[k]]
        x, y = a0, b0
        mx = 0.0
        while x < a1 and y < b1:
            if ids[x] == ids[y]:
                w = idf[ids[x]]
                wi += w
                ci += 1
                if w > mx:
                    mx = w
                x += 1
                y += 1
            elif ids[x] < ids[y]:
                x += 1
            else:
                y += 1
        out[i, 0] = wi / wa if wa > 0 else 0.0
        out[i, 1] = wi / wb if wb > 0 else 0.0
        u = wa + wb - wi
        out[i, 2] = wi / u if u > 0 else 0.0
        out[i, 3] = ci
        out[i, 4] = mx
        out[i, 5] = wa - wi  # idf mass of S1 tokens missing in pool record
    return out


@nb.njit(cache=True)
def _digits(v, buf):
    n = 0
    if v == 0:
        buf[0] = 0
        return 1
    while v > 0 and n < 20:
        buf[n] = v % 10
        v //= 10
        n += 1
    return n


@nb.njit(cache=True)
def _dlev(a, b):
    """Levenshtein distance between the decimal digit strings of a and b."""
    da = np.zeros(20, np.int64)
    db = np.zeros(20, np.int64)
    na = _digits(a, da)
    nb_ = _digits(b, db)
    prev = np.arange(nb_ + 1)
    cur = np.zeros(nb_ + 1, np.int64)
    for i in range(1, na + 1):
        cur[0] = i
        for j in range(1, nb_ + 1):
            c = 0 if da[i - 1] == db[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + c)
        for j in range(nb_ + 1):
            prev[j] = cur[j]
    return prev[nb_]


@nb.njit(cache=True)
def _ndig(v):
    n = 1
    v = abs(v)
    while v >= 10:
        v //= 10
        n += 1
    return n


@nb.njit(parallel=True, cache=True)
def _numfeat(ptr, vals, first, ra, rb):
    """Number agreement. Decoys ('branches') sit at a nearby house number with the same digit
    count, while noise on true records drops / changes single digits -> both the digit edit
    distance and the numeric difference / digit-length difference are exposed."""
    n = ra.shape[0]
    out = np.full((n, 11), -1.0, dtype=np.float32)
    for i in nb.prange(n):
        a0, a1 = ptr[ra[i]], ptr[ra[i] + 1]
        b0, b1 = ptr[rb[i]], ptr[rb[i] + 1]
        na, nbb = a1 - a0, b1 - b0
        out[i, 0] = na
        out[i, 1] = nbb
        if na == 0 or nbb == 0:
            continue
        x, y, c = a0, b0, 0
        while x < a1 and y < b1:
            if vals[x] == vals[y]:
                c += 1
                x += 1
                y += 1
            elif vals[x] < vals[y]:
                x += 1
            else:
                y += 1
        out[i, 2] = c / na
        out[i, 3] = c / nbb
        fa, fb = first[ra[i]], first[rb[i]]
        out[i, 4] = 1.0 if fa == fb else 0.0
        # closeness of the S1 first number to ANY pool number (digit edit distance)
        best = 99
        for y in range(b0, b1):
            d = _dlev(fa, vals[y])
            if d < best:
                best = d
        out[i, 5] = best
        # is the S1 first number present anywhere in the pool record
        hit = 0.0
        for y in range(b0, b1):
            if vals[y] == fa:
                hit = 1.0
        out[i, 6] = hit
        # numerically closest pool number: absolute difference and digit-length difference
        bd = 1 << 60
        bv = 0
        for y in range(b0, b1):
            d = abs(vals[y] - fa)
            if d < bd:
                bd = d
                bv = vals[y]
        out[i, 7] = min(bd, 1000000)
        out[i, 8] = _ndig(fa) - _ndig(bv)
        if fa >= 0 and fb >= 0:
            out[i, 9] = min(abs(fa - fb), 1000000)
            out[i, 10] = _ndig(fa) - _ndig(fb)
    return out


# ============================================================================ feature builder
RF = [
    ("n_ratio", "name_core", fuzz.ratio),
    ("n_tsort", "name_core", fuzz.token_sort_ratio),
    ("n_tset", "name_core", fuzz.token_set_ratio),
    ("n_partial", "name_core", fuzz.partial_ratio),
    ("n_jw", "name_core", JaroWinkler.normalized_similarity),
    ("n_join", "name_join", fuzz.ratio),
    ("n_join_partial", "name_join", fuzz.partial_ratio),
    ("n_full", "name_full", fuzz.ratio),
    ("n_sorted_lev", "name_sorted", Levenshtein.normalized_similarity),
    ("a_ratio", "addr_full", fuzz.ratio),
    ("a_tsort", "addr_full", fuzz.token_sort_ratio),
    ("a_tset", "addr_full", fuzz.token_set_ratio),
    ("a_partial", "addr_full", fuzz.partial_ratio),
    ("ac_ratio", "addr_core", fuzz.ratio),
    ("ac_tsort", "addr_core", fuzz.token_sort_ratio),
    ("ac_tset", "addr_core", fuzz.token_set_ratio),
]


def pair_features(store: Store, s1_row: np.ndarray, p_row: np.ndarray) -> pl.DataFrame:
    """Row-aligned features for pairs (s1_row[i], p_row[i])."""
    ra = s1_row.astype(np.int64)
    rbp = p_row.astype(np.int64)
    rb = rbp + store.n1  # index into the concatenated (S1 + pool) arrays
    cols = {}
    s1 = store.s1
    pool = store.pool
    cache = {}
    for name, col, scorer in RF:
        if col not in cache:
            cache[col] = (s1[col].gather(ra).to_list(), pool[col].gather(rbp).to_list())
        a, b = cache[col]
        cols[name] = process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
    del cache
    ov = _overlap(store.name_ptr, store.name_ids, store.name_idf, ra, rb)
    for j, nm in enumerate(["n_w1", "n_w2", "n_wj", "n_common", "n_maxidf", "n_miss"]):
        cols[nm] = ov[:, j]
    ov = _overlap(store.addr_ptr, store.addr_ids, store.addr_idf, ra, rb)
    for j, nm in enumerate(["a_w1", "a_w2", "a_wj", "a_common", "a_maxidf", "a_miss"]):
        cols[nm] = ov[:, j]
    nf = _numfeat(store.num_ptr, store.num_vals, store.num_first, ra, rb)
    for j, nm in enumerate(["num_n1", "num_n2", "num_f1", "num_f2", "num_first_eq", "num_first_dlev", "num_first_in",
                            "num_absdiff", "num_lendiff", "num_first_absdiff", "num_first_lendiff"]):
        cols[nm] = nf[:, j]
    # record-level attributes
    cols["p_is_domain"] = pool["is_domain"].gather(rbp).to_numpy().astype(np.float32)
    cols["p_has_alias"] = pool["has_alias"].gather(rbp).to_numpy().astype(np.float32)
    cols["p_name_indic"] = pool["name_indic"].gather(rbp).to_numpy().astype(np.float32)
    cols["p_addr_empty"] = pool["addr_empty"].gather(rbp).to_numpy().astype(np.float32)
    cols["p_is_s3"] = pool["entity_id"].gather(rbp).str.starts_with("S3").to_numpy().astype(np.float32)
    la = s1["legal"].gather(ra)
    lb = pool["legal"].gather(rbp)
    cols["legal_eq"] = ((la == lb) & (la != "")).to_numpy().astype(np.float32)
    cols["legal_s1_empty"] = (la == "").to_numpy().astype(np.float32)
    cols["legal_p_empty"] = (lb == "").to_numpy().astype(np.float32)
    rx = r"\b(" + "|".join(MARKERS_TRAIN) + r")\b"
    pm = pool["name_full"].gather(rbp).str.extract_all(rx)
    sm = s1["name_full"].gather(ra).str.extract_all(rx)
    cols["p_marker_extra"] = pl.DataFrame({"a": pm, "b": sm}).select(pl.col("a").list.set_difference(pl.col("b")).list.len()).to_series().to_numpy().astype(np.float32)
    # acronym names, domains keeping the legal form, generated random names
    pn = pool["name_core"].gather(rbp)
    single = (~pn.str.contains(" ")) & pn.str.len_chars().is_between(2, 6)
    cols["n_acronym"] = (single & ((pn == s1["init_core"].gather(ra)) | (pn == s1["init_full"].gather(ra)))).to_numpy().astype(np.float32)
    cols["n_join_full"] = process.cpdist(pool["name_join"].gather(rbp).to_list(), s1["full_join"].gather(ra).to_list(),
                                         scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    cols["p_name_unseen"] = store.p_unseen[rbp]
    cols["name_fp_p"] = store.name_fp[rb]
    cols["addr_fp_p"] = store.addr_fp[rb]
    cols["name_f1_s1"] = store.name_f1[ra]
    cols["name_fp_s1"] = store.name_fp[ra]
    cols["name_f1_p"] = store.name_f1[rb]
    cols["addr_f1_s1"] = store.addr_f1[ra]
    cols["addr_fp_s1"] = store.addr_fp[ra]
    cols["addr_f1_p"] = store.addr_f1[rb]
    cols["n_len1"] = s1["name_core"].gather(ra).str.len_chars().to_numpy().astype(np.float32)
    cols["n_len2"] = pool["name_core"].gather(rbp).str.len_chars().to_numpy().astype(np.float32)
    cols["a_len1"] = s1["addr_full"].gather(ra).str.len_chars().to_numpy().astype(np.float32)
    cols["a_len2"] = pool["addr_full"].gather(rbp).str.len_chars().to_numpy().astype(np.float32)
    return pl.DataFrame(cols)


def context_features(cand: pl.DataFrame, score_col: str, prefix: str) -> pl.DataFrame:
    """Competition features of a pair score within the S1's list and the pool record's list."""
    s = pl.col(score_col)
    gp = cand.group_by("p_row").agg(s.max().alias("_pmax"), s.top_k(2).min().alias("_p2"), pl.len().alias("_np"))
    gs = cand.group_by("s1_row").agg(s.max().alias("_smax"), pl.len().alias("_ns"))
    cand = cand.join(gp, on="p_row", how="left").join(gs, on="s1_row", how="left")
    cand = cand.with_columns(
        s.rank("ordinal", descending=True).over("s1_row").cast(pl.Float32).alias(f"{prefix}_rank_s1"),
        s.rank("ordinal", descending=True).over("p_row").cast(pl.Float32).alias(f"{prefix}_rank_p"),
    )
    second = pl.when(pl.col("_np") >= 2).then(pl.col("_p2")).otherwise(0.0)
    cand = cand.with_columns(
        pl.col("_ns").cast(pl.Float32).alias(f"{prefix}_n_s1"),
        pl.col("_np").cast(pl.Float32).alias(f"{prefix}_n_p"),
        (s - pl.col("_pmax")).alias(f"{prefix}_gap_pmax"),
        (s - pl.col("_smax")).alias(f"{prefix}_gap_s1max"),
        (s - pl.when(pl.col(f"{prefix}_rank_p") == 1).then(second).otherwise(pl.col("_pmax"))).alias(f"{prefix}_margin_p"),
    )
    return cand.drop("_pmax", "_p2", "_np", "_smax", "_ns")
