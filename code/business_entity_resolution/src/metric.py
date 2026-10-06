"""Macro F0.5 exactly as described in the challenge: per Source-1 entity, then averaged.

* true singleton + empty prediction -> 1.0 ; true singleton + any prediction -> 0.0
* entity with matches + empty prediction -> 0.0
"""
import polars as pl


def f05_macro(pred: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series, beta: float = 0.5, return_per_entity=False):
    """pred, gt: DataFrames with columns (s1, p) - one row per matched pair.
    s1_ids: every Source-1 entity in the evaluation set."""
    b2 = beta * beta
    base = pl.DataFrame({"s1": s1_ids}).unique()
    npred = pred.group_by("s1").agg(pl.len().alias("npred"))
    ngt = gt.group_by("s1").agg(pl.len().alias("ngt"))
    tp = pred.join(gt, on=["s1", "p"], how="inner").group_by("s1").agg(pl.len().alias("tp"))
    d = (base.join(npred, on="s1", how="left").join(ngt, on="s1", how="left").join(tp, on="s1", how="left")
             .fill_null(0))
    prec = pl.col("tp") / pl.col("npred")
    rec = pl.col("tp") / pl.col("ngt")
    f = (pl.when((pl.col("ngt") == 0) & (pl.col("npred") == 0)).then(1.0)
           .when((pl.col("ngt") == 0) | (pl.col("npred") == 0) | (pl.col("tp") == 0)).then(0.0)
           .otherwise((1 + b2) * prec * rec / (b2 * prec + rec)))
    d = d.with_columns(f.alias("f"))
    if return_per_entity:
        return d
    return d["f"].mean()
