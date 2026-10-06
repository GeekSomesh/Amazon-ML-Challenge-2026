"""Direction of the house-number change (pool - S1) for same-name pairs:
  up_same_len : pool number larger, same digit count  (planted look-alike 'a few doors up')
  down_same_len, len_change (digit dropped/added)      (per-record / base noise)
(a) validation labels (natural records only, US/India); (b) France pairs removed by a leaderboard probe that showed
them to be correct; (d) France pairs added when the France recalibration is switched off.

  python offset_dir.py <baseline_dir> <probe_removed_dir> <probe_added_dir>
Directories hold a matching_results.tsv; relative names are resolved under BER_SUBMISSIONS
(default: a 'submissions' folder next to the work directory)."""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import os
import sys

import polars as pl

from config import WORK, norm_path

SUB = os.environ.get("BER_SUBMISSIONS", os.path.join(os.path.dirname(os.path.abspath(WORK)), "submissions"))


def cls(n1, n2):
    return (pl.when(n1.is_null() | n2.is_null()).then(pl.lit("missing"))
              .when(n1 == n2).then(pl.lit("equal"))
              .when(n1.cast(pl.Utf8).str.len_chars() != n2.cast(pl.Utf8).str.len_chars()).then(pl.lit("len_change"))
              .when(n2 > n1).then(pl.lit("up_same_len")).otherwise(pl.lit("down_same_len")))


def num(c):
    return pl.col(c).str.replace_all(r"(\d)\s*-\s*(\d)", "$1$2").str.extract(r"(\d+)", 1).cast(pl.Int64, strict=False)


def load_pairs(v):
    d = v if os.path.isdir(v) else os.path.join(SUB, v)
    m = pl.read_csv(os.path.join(d, "matching_results.tsv"), separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
    return (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "p"}))


def main(baseline, probe_removed, probe_added):
    t1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet")).rename({"entity_id": "s1"})
    tp = pl.concat([pl.read_parquet(os.path.join(WORK, f"test_s{s}.parquet"), columns=["entity_id", "business_address"]) for s in (2, 3)]).rename(
        {"entity_id": "p", "business_address": "paddr"})
    fr = t1.filter(pl.col("country") == "France")
    # (b) France pairs removed by the probe (shown to be correct by the leaderboard)
    base, removed = load_pairs(baseline), load_pairs(probe_removed)
    b = base.join(removed, on=["s1", "p"], how="anti").join(fr, on="s1").join(tp, on="p")
    # (d) France pairs added by the probe
    added = load_pairs(probe_added)
    d = added.join(base, on=["s1", "p"], how="anti").join(fr, on="s1").join(tp, on="p")
    for name, x in [("(b) France pairs removed by the probe [correct]", b), ("(d) France pairs added by the probe", d)]:
        x = x.with_columns(cls(num("business_address"), num("paddr")).alias("dir"))
        print(name, x.height, x.group_by("dir").len().with_columns((pl.col("len") / x.height).round(3).alias("share")).sort("dir").rows(), flush=True)
    # (a) validation labels: same-name deviating-number pairs, natural records only
    from eval_oof import load_gt_rows
    s1, gt = load_gt_rows("trainsim")
    oof = pl.scan_parquet(os.path.join(WORK, "oof_s12ens.parquet")).select("s1_row", "p_row", "label", "prob").filter(pl.col("prob") >= 0.05).collect(engine="streaming")
    a1 = pl.read_parquet(norm_path("trainsim", 1), columns=["addr_full", "name_core"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
    ap = pl.concat([pl.read_parquet(norm_path("trainsim", s), columns=["entity_id", "addr_full", "name_core"]) for s in (2, 3)]).with_row_index("p_row").with_columns(pl.col("p_row").cast(pl.UInt32))
    x = (oof.join(a1, on="s1_row").join(ap, on="p_row", suffix="_p").filter(~pl.col("entity_id").str.contains("DCY") & (pl.col("name_core") == pl.col("name_core_p")))
            .with_columns(cls(pl.col("addr_full").str.extract(r"\b(\d+)\b", 1).cast(pl.Int64, strict=False),
                              pl.col("addr_full_p").str.extract(r"\b(\d+)\b", 1).cast(pl.Int64, strict=False)).alias("dir")))
    print("(a) VALIDATION natural same-name pairs (prob>=0.05): true rate by direction")
    print(x.group_by("dir").agg(pl.len(), pl.col("label").mean().round(3).alias("true_rate"), pl.col("prob").mean().round(3).alias("mean_p")).sort("dir").rows(), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
