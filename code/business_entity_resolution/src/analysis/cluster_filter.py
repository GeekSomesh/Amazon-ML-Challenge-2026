"""One-base-per-source cluster consistency (decision-layer post-filter).

Training data: the generator derives ONE noisy base record per source for each business and copies
it (in 91.8% of (entity, source) groups with >=2 true records, all records carry the same house
number). So if the records selected for one S1 entity from one source split into several house-number
groups, the extra groups are planted decoys (a different business a few doors away). Numbers that are
digit-deletion / leading-zero variants of each other count as the same group (per-record noise).
We keep, per (S1, source), only the group with the highest selection mass; number-less records are kept.
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import numpy as np
import polars as pl


def _is_variant(a: str, b: str) -> bool:
    """b obtained from a by deleting one digit or leading zeros (or equal)."""
    a, b = a.lstrip("0") or "0", b.lstrip("0") or "0"
    if a == b:
        return True
    if len(a) == len(b) + 1:
        return any(a[:i] + a[i + 1:] == b for i in range(len(a)))
    if len(b) == len(a) + 1:
        return any(b[:i] + b[i + 1:] == a for i in range(len(b)))
    return False


def cluster_filter(sel: pl.DataFrame, num_first: np.ndarray, p_src: pl.Series, n1: int, col="q",
                   s1_in_p: np.ndarray = None) -> pl.DataFrame:
    """sel: selected pairs with columns s1_row, p_row, <col>. num_first: first house number per record of the
    concatenated (S1 + pool) arrays (-1 = none). p_src: 'S2'/'S3' per pool row. s1_in_p (aligned with sel):
    1 if the S1's first number occurs anywhere in the pool record - such records are assigned to the S1's
    own number group (robust to component reordering). Returns the filtered selection."""
    num = num_first[sel["p_row"].to_numpy().astype(np.int64) + n1].copy()
    if s1_in_p is not None:
        s1num = num_first[sel["s1_row"].to_numpy().astype(np.int64)]
        num = np.where(s1_in_p == 1, s1num, num)
    d = sel.with_columns(pl.Series("_num", num), pl.Series("_src", p_src.gather(sel["p_row"].to_numpy())))
    # only (S1, source) groups with >= 2 distinct numbers need work
    multi = (d.filter(pl.col("_num") >= 0).group_by("s1_row", "_src").agg(pl.col("_num").n_unique().alias("_k"))
               .filter(pl.col("_k") >= 2).select("s1_row", "_src"))
    work = d.join(multi, on=["s1_row", "_src"], how="semi").filter(pl.col("_num") >= 0)
    drop = []
    for (s1, src), g in work.group_by(["s1_row", "_src"]):
        nums = g["_num"].to_list()
        qs = g[col].to_list()
        prs = g["p_row"].to_list()
        # union digit-variant numbers into groups
        uniq = sorted(set(nums))
        parent = {u: u for u in uniq}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        for i, a in enumerate(uniq):
            for b in uniq[i + 1:]:
                if _is_variant(str(a), str(b)):
                    parent[find(a)] = find(b)
        mass = {}
        for n, q in zip(nums, qs):
            mass[find(n)] = mass.get(find(n), 0.0) + q
        if len(mass) < 2:
            continue
        best = max(mass, key=mass.get)
        drop += [(s1, p) for n, p in zip(nums, prs) if find(n) != best]
    if not drop:
        return sel
    dd = pl.DataFrame({"s1_row": [a for a, _ in drop], "p_row": [b for _, b in drop]},
                      schema={"s1_row": sel.schema["s1_row"], "p_row": sel.schema["p_row"]})
    print(f"cluster filter: dropped {dd.height} of {sel.height} selected pairs (extra same-source number groups)")
    return sel.join(dd, on=["s1_row", "p_row"], how="anti")
