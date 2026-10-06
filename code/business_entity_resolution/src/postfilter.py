"""Decoy-marker post-filter.

EDA on training labels: the generator plants 'sister company' / 'branch' decoys whose names
carry a marker word that the real entity does not have (US: holdings, westgate, riverside,
southside, midtown, lakeside, northside, eastgate, greater, central, uptown, downtown;
India: holdings, infratech, overseas). Such records are true matches 0-14% of the time,
and every selected pair carrying one on validation was a false positive.
A selected pair is dropped when the pool name contains a marker the S1 name lacks.
"""
import re

import polars as pl

MARKERS_TRAIN = ["holdings", "westgate", "riverside", "southside", "midtown", "lakeside", "northside",
                 "eastgate", "greater", "central", "uptown", "downtown", "infratech", "overseas"]
# French sister-company words found unsupervised (5-1700x more frequent in test S2/S3 France names than
# in S1 France names). NOT used: a public-leaderboard A/B comparison (0.97268 with this filter vs 0.97320
# without) showed that they also occur on true French matches, unlike the English branch / holdings words.
MARKERS_FR = ["holding", "groupe", "developpement", "participations", "distribution", "international"]


def drop_marker_pairs(sel: pl.DataFrame, s1_names: pl.Series, pool_names: pl.Series, markers) -> pl.DataFrame:
    """sel: (s1_row, p_row) selected pairs; *_names: normalised name_full per row."""
    rx = r"\b(" + "|".join(re.escape(m) for m in markers) + r")\b"
    d = sel.with_columns(pl.Series("_pn", pool_names.gather(sel["p_row"].to_numpy())),
                         pl.Series("_sn", s1_names.gather(sel["s1_row"].to_numpy())))
    d = d.with_columns(pl.col("_pn").str.extract_all(rx).alias("_pm"), pl.col("_sn").str.extract_all(rx).alias("_sm"))
    d = d.with_columns(pl.col("_pm").list.set_difference(pl.col("_sm")).list.len().alias("_extra"))
    dropped = d.filter(pl.col("_extra") > 0).height
    print(f"post-filter: dropped {dropped} of {sel.height} selected pairs carrying decoy markers")
    return d.filter(pl.col("_extra") == 0).select(sel.columns)
