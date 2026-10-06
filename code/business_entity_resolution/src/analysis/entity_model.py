"""Entity-level model for the decision layer: from the per-S1 distribution of one-owner probabilities q,
predict P(S1 has no true match) and the expected number of true matches; feed both into the expected-F0.5
selection (instead of prod(1-q) and sum(q)). Cross-fitted by the same S1-entity folds.

  python entity_model.py <oof_tag>          -> validation F0.5: baseline vs entity-model decision
  (test use: apply_entity(test_scored, models) in make_submission)
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # pipeline modules in src/

import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from config import N_THREADS, WORK
from decide import B2, evaluate, one_owner

TOP = 6
SUB = int(os.environ.get("ENT_SUB", "1"))


def entity_features(oo: pl.DataFrame) -> pl.DataFrame:
    x = oo.filter(pl.col("q") >= 0.005).sort(["s1_row", "q"], descending=[False, True])
    g = x.group_by("s1_row").agg(
        [pl.col("q").get(i, null_on_oob=True).alias(f"q{i + 1}") for i in range(TOP)] + [
            pl.col("q").sum().alias("qsum"), (pl.col("q") > 0.5).sum().alias("n05"), (pl.col("q") > 0.1).sum().alias("n01"),
            (pl.col("q") > 0.9).sum().alias("n09"), (1 - pl.col("q")).clip(1e-6, 1).log().sum().alias("logp0"), pl.len().alias("ncand")])
    return g.with_columns([pl.col(f"q{i + 1}").fill_null(0.0) for i in range(TOP)])


def select_with_entity(oo: pl.DataFrame, ent: pl.DataFrame, min_q=0.02):
    """expected-F prefix selection with learned E|G| (n_hat) and P(empty) (p_empty)."""
    x = oo.filter(pl.col("q") >= min_q).sort(["s1_row", "q"], descending=[False, True]).join(ent, on="s1_row", how="left")
    x = x.with_columns(pl.col("q").cum_sum().over("s1_row").alias("_cs"), pl.int_range(1, pl.len() + 1).over("s1_row").alias("_k"))
    x = x.with_columns(((1 + B2) * pl.col("_cs") / (B2 * pl.col("n_hat").clip(0.3, None) + pl.col("_k"))).alias("_ef"))
    best = x.group_by("s1_row").agg(pl.col("_ef").max().alias("_best"), pl.col("p_empty").first())
    x = x.join(best, on="s1_row")
    kb = x.filter(pl.col("_ef") == pl.col("_best")).group_by("s1_row").agg(pl.col("_k").min().alias("_kb"))
    x = x.join(kb, on="s1_row").filter((pl.col("_k") <= pl.col("_kb")) & (pl.col("_best") > pl.col("p_empty")))
    return x.select("s1_row", "p_row")


def main(tag):
    from eval_oof import load_gt_rows
    from decide import select_expected_f
    oof = (pl.scan_parquet(os.path.join(WORK, f"oof_{tag}.parquet")).select("s1_row", "p_row", "fold", "prob")
             .filter(pl.col("prob") >= 0.002).filter(pl.col("s1_row") % SUB == 0).collect(engine="streaming"))
    s1, gt = load_gt_rows("trainsim")
    # evaluation on a subsample of S1 entities (memory); baseline and variants use the same subsample
    s1 = s1.filter(pl.col("s1_row") % SUB == 0)
    gt = gt.filter(pl.col("s1_row") % SUB == 0)
    rows = s1["s1_row"].cast(pl.UInt32).to_numpy()
    oo = one_owner(oof.select("s1_row", "p_row", "prob"), "prob", "q")
    base = evaluate(select_expected_f(oo, "q"), gt, rows)
    ef = entity_features(oo)
    ngt = gt.group_by("s1_row").len().rename({"len": "ngt"})
    folds = oof.select("s1_row", "fold").unique("s1_row")
    ent = (s1.select(pl.col("s1_row").cast(pl.UInt32), "country").join(folds, on="s1_row", how="left")
             .join(ef, on="s1_row", how="left").join(ngt, on="s1_row", how="left")
             .with_columns(pl.col("ngt").fill_null(0), pl.col("fold").fill_null(0)))
    ent = ent.with_columns([pl.col(c).fill_null(0.0) for c in ef.columns if c != "s1_row"])
    feats = [c for c in ef.columns if c != "s1_row"] + ["is_us"]
    ent = ent.with_columns((pl.col("country") == "US").cast(pl.Float32).alias("is_us"))
    p_empty = np.zeros(ent.height)
    n_hat = np.zeros(ent.height)
    P = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9, verbose=-1, num_threads=N_THREADS, seed=3)
    for f in (0, 1):
        tr = ent.filter(pl.col("fold") != f)
        te = (ent["fold"] == f).to_numpy()
        X, Xt = tr.select(feats).to_numpy().astype(np.float32), ent.filter(pl.col("fold") == f).select(feats).to_numpy().astype(np.float32)
        m1 = lgb.train(dict(P, objective="binary"), lgb.Dataset(X, (tr["ngt"] == 0).to_numpy().astype(float)), 400)
        m2 = lgb.train(dict(P, objective="poisson"), lgb.Dataset(X, tr["ngt"].to_numpy().astype(float)), 400)
        p_empty[te] = m1.predict(Xt)
        n_hat[te] = m2.predict(Xt)
        m1.save_model(os.path.join(WORK, f"ent_{tag}_empty_fold{f}.txt"))
        m2.save_model(os.path.join(WORK, f"ent_{tag}_count_fold{f}.txt"))
    e = ent.select("s1_row").with_columns(pl.Series("p_empty", p_empty), pl.Series("n_hat", n_hat))
    print(f"[{tag}] baseline owner+expF F0.5 = {base:.5f}", flush=True)
    for name, ee in [("learned p_empty + n_hat", e),
                     ("learned n_hat only", e.join(ef.select("s1_row", "logp0"), on="s1_row", how="left")
                          .with_columns(pl.col("logp0").fill_null(0.0).exp().alias("p_empty")).drop("logp0")),
                     ("learned p_empty only", e.join(ef.select("s1_row", "qsum"), on="s1_row", how="left")
                          .with_columns(pl.col("qsum").fill_null(0.0).alias("n_hat")).drop("qsum"))]:
        f05 = evaluate(select_with_entity(oo, ee), gt, rows)
        print(f"   {name:26s}: {f05:.5f} ({f05 - base:+.5f})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
