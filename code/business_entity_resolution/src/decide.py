"""Decision layer: pair probabilities -> final match lists.

1. exclusivity: every Source-2/3 record belongs to at most one Source-1 entity (verified
   in training: 0 records appear in two lists) -> keep only its best-scoring S1.
2. per-S1 subset choice:
   (a) global threshold on the probability, or
   (b) expected-F0.5 maximisation: sort the S1's candidates by probability q, pick the
       prefix k maximising  E[F] ~ (1+b^2) * sum_{j<=k} q_j / (b^2 * E|G| + k)
       and compare it with predicting nothing (worth P(no match) = prod(1-q_j)).
"""
import numpy as np
import polars as pl

from metric import f05_macro

B2 = 0.25


def exclusive(pairs: pl.DataFrame, col="prob", margin=0.0):
    """Keep, for each pool record, only its best S1 (optionally only if it beats the runner-up by `margin`)."""
    d = pairs.with_columns(
        pl.col(col).rank("ordinal", descending=True).over("p_row").alias("_r"),
        pl.col(col).top_k(2).over("p_row", mapping_strategy="join").alias("_top2"),
    )
    d = d.with_columns((pl.col(col) - pl.col("_top2").list.get(1, null_on_oob=True).fill_null(0.0)).alias("excl_margin"))
    d = d.filter(pl.col("_r") == 1)
    if margin > 0:
        d = d.filter(pl.col("excl_margin") >= margin)
    return d.drop("_r", "_top2")


def one_owner(pairs: pl.DataFrame, col="prob", out="q"):
    """Soft exclusivity: a pool record has at most one owner. Treat each candidate S1 as an
    option with odds o = p/(1-p) against a 'no owner' option of odds 1:
        q_i = o_i / (1 + sum_j o_j)
    then keep only the best owner of each pool record (hard exclusivity) with probability q."""
    o = (pl.col(col).clip(1e-6, 1 - 1e-6) / (1 - pl.col(col).clip(1e-6, 1 - 1e-6)))
    d = pairs.with_columns(o.alias("_o"))
    d = d.with_columns((pl.col("_o") / (1 + pl.col("_o").sum().over("p_row"))).alias(out))
    d = d.with_columns(pl.col(out).rank("ordinal", descending=True).over("p_row").alias("_r"))
    return d.filter(pl.col("_r") == 1).drop("_o", "_r")


def select_threshold(d: pl.DataFrame, t, col="prob"):
    return d.filter(pl.col(col) >= t).select("s1_row", "p_row")


def select_expected_f(d: pl.DataFrame, col="prob", eg_col=None, min_q=0.02):
    """d: pairs after exclusivity. eg_col: optional per-S1 expected |G| (else sum of q)."""
    x = d.filter(pl.col(col) >= min_q).sort(["s1_row", col], descending=[False, True])
    x = x.with_columns(
        pl.col(col).cum_sum().over("s1_row").alias("_cs"),
        pl.int_range(1, pl.len() + 1).over("s1_row").alias("_k"),
        pl.col(col).sum().over("s1_row").alias("_eg") if eg_col is None else pl.col(eg_col).alias("_eg"),
        (1 - pl.col(col)).clip(1e-6, 1).log().sum().over("s1_row").exp().alias("_p0"),
    )
    x = x.with_columns(((1 + B2) * pl.col("_cs") / (B2 * pl.col("_eg") + pl.col("_k"))).alias("_ef"))
    best = x.group_by("s1_row").agg(pl.col("_ef").max().alias("_best"), pl.col("_p0").first())
    x = x.join(best, on="s1_row")
    kbest = x.filter(pl.col("_ef") == pl.col("_best")).group_by("s1_row").agg(pl.col("_k").min().alias("_kb"))
    x = x.join(kbest, on="s1_row").filter((pl.col("_k") <= pl.col("_kb")) & (pl.col("_best") > pl.col("_p0")))
    return x.select("s1_row", "p_row")


def evaluate(sel: pl.DataFrame, gt: pl.DataFrame, s1_rows: np.ndarray):
    return f05_macro(sel.rename({"s1_row": "s1", "p_row": "p"}),
                     gt.rename({"s1_row": "s1", "p_row": "p"}), pl.Series("s1", s1_rows))
