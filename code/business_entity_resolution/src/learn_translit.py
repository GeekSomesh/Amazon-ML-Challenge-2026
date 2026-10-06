"""Learn Indic-script -> Latin dictionaries from TRAINING ground-truth pairs only.

EDA showed that when a Source-2/3 name is written in an Indic script, it is a
word-by-word transliteration of the English name (token counts agree 99.999%),
so aligning tokens by position and taking a majority vote yields an exact
dictionary (e.g. "प्राइवेट" -> "private", "प्रा." -> "pvt").
Native-script address fragments are almost always the state name, so for each
native comma-segment we pick the most frequent co-occurring Source-1 segment.
Output: work/translit.json  {"name": {native_token: latin_token},
                              "addr": {native_segment: latin_segment}}
"""
import collections
import json
import os
import re

import polars as pl

from config import WORK

INDIC = re.compile(r"[ऀ-෿]")


def latin_tokens(s):
    return re.findall(r"\w+", s.lower())


def main():
    s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"))
    pool = pl.concat([pl.read_parquet(os.path.join(WORK, "train_s2.parquet")),
                      pl.read_parquet(os.path.join(WORK, "train_s3.parquet"))])
    gt = pl.read_parquet(os.path.join(WORK, "train_gt.parquet"))
    ex = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "")
            .rename({"source1_entity_id": "s1", "matched_entity_ids": "p"}))
    pairs = (ex.join(pool.rename({"entity_id": "p", "business_name": "pn", "business_address": "pa"}), on="p")
               .join(s1.rename({"entity_id": "s1", "business_name": "sn", "business_address": "sa"}), on="s1"))

    # ---- names: positional token alignment
    nm = pairs.filter(pl.col("pn").str.contains(r"[ऀ-෿]"))
    cnt = collections.defaultdict(collections.Counter)
    for sn, pn in zip(nm["sn"].to_list(), nm["pn"].to_list()):
        a = latin_tokens(sn)
        b = pn.split()
        if len(a) != len(b):
            continue
        for x, y in zip(b, a):
            if INDIC.search(x):
                cnt[x][y] += 1
                x2 = x.strip(".,()")
                if x2 != x:
                    cnt[x2][y] += 1
    name_map = {k: c.most_common(1)[0][0] for k, c in cnt.items() if sum(c.values()) >= 1}
    print("name dictionary size:", len(name_map))

    # ---- addresses: native comma-segment -> most frequent co-occurring S1 segment
    ad = pairs.filter(pl.col("pa").str.contains(r"[ऀ-෿]"))
    acnt = collections.defaultdict(collections.Counter)
    for sa, pa in zip(ad["sa"].to_list(), ad["pa"].to_list()):
        segs1 = {s.strip().lower() for s in sa.split(",") if s.strip()}
        for seg in pa.split(","):
            seg = seg.strip()
            if INDIC.search(seg):
                for s in segs1:
                    acnt[seg][s] += 1
    addr_map = {}
    for k, c in acnt.items():
        (best, n), = c.most_common(1)
        tot = sum(1 for _ in c)  # distinct
        addr_map[k] = best
    print("addr dictionary size:", len(addr_map))
    for k in list(addr_map)[:40]:
        print("   ", k, "->", addr_map[k], acnt[k].most_common(3))
    with open(os.path.join(WORK, "translit.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name_map, "addr": addr_map}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
