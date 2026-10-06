"""France-only decision filters applied to a finished submission (IDs level), for leaderboard probes and
for the final France post-filter.

  python france_filter.py <in_dir> <out_dir> <rule>[,<rule>...]
    numdiff : drop France pairs whose first house number differs from the S1's (both present)
    marker  : drop France pairs whose pool name adds a French look-alike marker word absent from the S1 name
Planted French look-alikes = brand + marker word (Groupe, Développement, International, Holding,
Participations, Distribution) at a house number a few doors away; genuine French copies keep the number.
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import os
import re
import shutil
import sys

import polars as pl

from config import WORK

MARKERS_FR = ["groupe", "developpement", "international", "holding", "participations", "distribution"]
MK = r"(?i)\b(" + "|".join(MARKERS_FR) + r")\b"


def fold(e):
    return e.str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()


def main(in_dir, out_dir, rules):
    rules = rules.split(",")
    m = pl.read_csv(os.path.join(in_dir, "matching_results.tsv"), separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
    pairs = (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
               .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "p"}))
    t1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet")).rename({"entity_id": "s1"})
    tp = pl.concat([pl.read_parquet(os.path.join(WORK, f"test_s{s}.parquet"), columns=["entity_id", "business_name", "business_address"])
                    for s in (2, 3)]).rename({"entity_id": "p", "business_name": "pname", "business_address": "paddr"})
    d = pairs.join(t1, on="s1").join(tp, on="p")
    fr = pl.col("country") == "France"
    num = lambda c: pl.col(c).str.replace_all(r"(\d)\s*-\s*(\d)", "$1$2").str.extract(r"(\d+)", 1).cast(pl.Int64, strict=False)
    drop = pl.lit(False)
    if "numdiff" in rules:
        drop = drop | (fr & num("business_address").is_not_null() & num("paddr").is_not_null() & (num("business_address") != num("paddr")))
    if "marker" in rules:
        drop = drop | (fr & fold(pl.col("pname")).str.contains(MK) & ~fold(pl.col("business_name")).str.contains(MK))
    d = d.with_columns(drop.alias("_drop"))
    print(f"rules {rules}: dropping {int(d['_drop'].sum())} of {d.filter(fr).height} France pairs "
          f"(France S1 affected: {d.filter(pl.col('_drop'))['s1'].n_unique()})", flush=True)
    keep = d.filter(~pl.col("_drop")).group_by("s1", maintain_order=True).agg(pl.col("p").sort().str.join(",").alias("matched_entity_ids"))
    out = m.select("source1_entity_id").join(keep.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left").with_columns(
        pl.col("matched_entity_ids").fill_null(""))
    os.makedirs(out_dir, exist_ok=True)
    out.write_csv(os.path.join(out_dir, "matching_results.tsv"), separator="\t", quote_style="never")
    shutil.copyfile(os.path.join(in_dir, "candidate_pairs.tsv"), os.path.join(out_dir, "candidate_pairs.tsv"))
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
