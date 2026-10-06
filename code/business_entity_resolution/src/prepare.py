"""Normalise every source file once and store the results as parquet.

work/{split}_s{src}.parquet  ->  work/{split}_s{src}_norm.parquet
"""
import os
import sys
import time

import polars as pl

from config import WORK
from text_norm import normalize_frame

KEEP = ["entity_id", "country", "name_full", "name_core", "name_join", "name_sorted", "legal",
        "is_domain", "has_alias", "name_indic", "addr_full", "addr_empty"]
CHUNK = 1_000_000


def run(split, src):
    t = time.time()
    df = pl.read_parquet(os.path.join(WORK, f"{split}_s{src}.parquet"))
    parts = []
    for off in range(0, df.height, CHUNK):
        parts.append(normalize_frame(df.slice(off, CHUNK)).select(KEEP))
    out = pl.concat(parts)
    out.write_parquet(os.path.join(WORK, f"{split}_s{src}_norm.parquet"))
    print(f"{split} s{src}: {out.height} rows in {time.time() - t:.0f}s", flush=True)


def strip_injected_address_tokens(split, min_ratio=5.0, min_fp=0.002):
    """Unsupervised, per country: address tokens that are >= min_ratio times more frequent in
    Source-2/3 than in Source-1 (the clean reference) were injected by the noise process
    (e.g. French department names where Source 1 names the region, 'N', 'door', 'region').
    They are removed from every record of the split. Uses only the split's own inputs."""
    paths = [os.path.join(WORK, f"{split}_s{s}_norm.parquet") for s in (1, 2, 3)]
    dfs = [pl.read_parquet(p) for p in paths]
    s1, pool = dfs[0], pl.concat(dfs[1:])

    def freq(df):
        return (df.select("country", pl.col("addr_full").str.split(" ").list.unique().alias("t")).explode("t")
                  .filter(pl.col("t").is_not_null() & (pl.col("t") != "")).group_by("country", "t").len())

    n1 = s1.group_by("country").len().rename({"len": "N1"})
    npl = pool.group_by("country").len().rename({"len": "NP"})
    f = (freq(pool).rename({"len": "cp"}).join(freq(s1).rename({"len": "c1"}), on=["country", "t"], how="left")
           .fill_null(0).join(n1, on="country").join(npl, on="country"))
    f = f.with_columns((pl.col("cp") / pl.col("NP")).alias("fp"), (pl.col("c1") / pl.col("N1")).alias("f1"))
    inj = f.filter((((pl.col("fp") + 1e-5) / (pl.col("f1") + 1e-5)) > min_ratio) & (pl.col("fp") > min_fp)
                   & ~pl.col("t").str.contains(r"^\d+$"))
    table = {(c[0] if isinstance(c, tuple) else c): g["t"].to_list() for c, g in inj.group_by("country")}
    for c, toks in sorted(table.items(), key=lambda kv: str(kv[0])):
        print(f"  [{split}] {c}: stripping injected address tokens {sorted(toks)}", flush=True)
    for p, df in zip(paths, dfs):
        e = pl.col("addr_full")
        for c, toks in table.items():
            e = pl.when(pl.col("country") == c).then(
                pl.col("addr_full").str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(toks))).list.join(" ")
            ).otherwise(e)
        df = df.with_columns(e.alias("addr_full")).with_columns((pl.col("addr_full").str.strip_chars() == "").alias("addr_empty"))
        df.write_parquet(p)


if __name__ == "__main__":
    splits = sys.argv[1:] or ["train", "test"]
    for split in splits:
        for src in (1, 2, 3):
            run(split, src)
        strip_injected_address_tokens(split)
