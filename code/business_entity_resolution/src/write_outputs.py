"""Write the two submission files in the exact required format.

matching_results.tsv : source1_entity_id \t matched_entity_ids   (comma-separated, may be empty)
candidate_pairs.tsv  : source1_entity_id \t candidate_entity_ids (the exact set the model scored)
Every Source-1 test entity gets exactly one row; IDs are unique within a list.
"""
import os

import polars as pl

from config import OUTPUT


def _write(s1_ids: pl.Series, pairs: pl.DataFrame, col: str, path: str):
    """pairs: (s1_id, p_id)"""
    agg = (pairs.unique(["s1_id", "p_id"]).sort(["s1_id", "p_id"])
                .group_by("s1_id", maintain_order=True).agg(pl.col("p_id").str.join(",").alias(col)))
    out = (pl.DataFrame({"source1_entity_id": s1_ids}).unique(maintain_order=True)
             .join(agg.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left")
             .with_columns(pl.col(col).fill_null("")))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for a, b in out.iter_rows():
            f.write(f"{a}\t{b}\n")
    return out.height


def write_submission(s1_ids: pl.Series, matches: pl.DataFrame, candidates: pl.DataFrame, out_dir=OUTPUT):
    os.makedirs(out_dir, exist_ok=True)
    n1 = _write(s1_ids, matches, "matched_entity_ids", os.path.join(out_dir, "matching_results.tsv"))
    n2 = _write(s1_ids, candidates, "candidate_entity_ids", os.path.join(out_dir, "candidate_pairs.tsv"))
    print(f"wrote {n1} matching rows ({matches.height} pairs) and {n2} candidate rows ({candidates.height} pairs) to {out_dir}")
