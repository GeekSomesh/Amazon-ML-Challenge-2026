"""Leaderboard probes that change ONLY one country's predictions relative to the final decision
(one-owner + expected-F + decoy-word filter), so the LB difference isolates that country.

  python make_probe.py <tag> <stage1_tag> <country> <mode> <arg> <out_dir>
    mode add  <q_min>: also select that country's pairs with q >= q_min (after one-owner)
    mode keep <q_min>: keep only that country's selected pairs with q >= q_min
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import os
import sys

import polars as pl

from config import WORK, norm_path
from decide import one_owner, select_expected_f
from make_submission import final_candidates
from postfilter import MARKERS_TRAIN, drop_marker_pairs
from write_outputs import write_submission


def main(tag, stage1_tag, country, mode, arg, out_dir):
    q_min = float(arg)
    sc = pl.read_parquet(os.path.join(WORK, f"test_scored_{tag}.parquet"), columns=["s1_row", "p_row", "prob"])
    sc = sc.join(final_candidates(sc, stage1_tag), on=["s1_row", "p_row"], how="semi")
    t1 = pl.read_parquet(norm_path("test", 1), columns=["entity_id", "name_full", "country"])
    tp = pl.concat([pl.read_parquet(norm_path("test", 2), columns=["entity_id", "name_full"]),
                    pl.read_parquet(norm_path("test", 3), columns=["entity_id", "name_full"])])
    oo = one_owner(sc, "prob", "q")
    base = drop_marker_pairs(select_expected_f(oo, "q"), t1["name_full"], tp["name_full"], MARKERS_TRAIN)
    rows_c = t1.with_row_index("s1_row").filter(pl.col("country") == country).select(pl.col("s1_row").cast(pl.UInt32))
    in_c = pl.col("s1_row").is_in(rows_c["s1_row"].implode())
    if mode == "add":
        extra = oo.filter(in_c & (pl.col("q") >= q_min)).select("s1_row", "p_row")
        extra = drop_marker_pairs(extra, t1["name_full"], tp["name_full"], MARKERS_TRAIN)
        sel = pl.concat([base, extra]).unique(["s1_row", "p_row"])
    elif mode == "keep":
        qq = base.join(oo.select("s1_row", "p_row", "q"), on=["s1_row", "p_row"])
        sel = qq.filter(~in_c | (pl.col("q") >= q_min)).select("s1_row", "p_row")
    else:
        raise ValueError(mode)
    changed = sel.join(base, on=["s1_row", "p_row"], how="anti").height, base.join(sel, on=["s1_row", "p_row"], how="anti").height
    qs = oo.join(pl.concat([sel.join(base, on=["s1_row", "p_row"], how="anti"), base.join(sel, on=["s1_row", "p_row"], how="anti")]),
                 on=["s1_row", "p_row"], how="semi")["q"]
    print(f"{country} {mode} {q_min}: added {changed[0]} pairs, removed {changed[1]} pairs; mean q of changed pairs {(qs.mean() or 0.0):.3f}; "
          f"affected S1 {pl.concat([sel, base]).join(rows_c, on='s1_row', how='semi').unique('s1_row').height}", flush=True)
    ids = lambda d: d.select(pl.Series("s1_id", t1["entity_id"].gather(d["s1_row"].to_numpy())),
                             pl.Series("p_id", tp["entity_id"].gather(d["p_row"].to_numpy())))
    os.makedirs(out_dir, exist_ok=True)
    write_submission(t1["entity_id"], ids(sel), ids(sc), out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:7])
