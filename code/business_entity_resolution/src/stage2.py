"""Stage-2 re-scorer: adds group / competition / sibling-agreement features on top of
stage-1 probabilities and trains a second LightGBM (same S1-entity folds).

Sibling agreement: the Source-2/3 records of one entity share a source-specific base
form (EDA: e.g. all S3 copies carry "##110 Meadowbrook Acres"), so a record that is
very similar to the *other confident candidates* of the same S1 entity is likely a
true match even when its own name is garbled or replaced; a co-located distractor
typically is not.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from features import Store

MIN_P1 = 0.005     # pairs below this stage-1 probability are not re-scored
TOPQ = 3           # number of confident siblings compared per S1


def group_features(c: pl.DataFrame) -> pl.DataFrame:
    """c: s1_row, p_row, prob1 (all candidate pairs)."""
    p = pl.col("prob1")
    c = c.with_columns(
        p.rank("ordinal", descending=True).over("p_row").cast(pl.Float32).alias("g_rank_p"),
        p.rank("ordinal", descending=True).over("s1_row").cast(pl.Float32).alias("g_rank_s1"),
        p.max().over("p_row").alias("g_pmax"),
        p.sum().over("p_row").alias("g_psum"),
        p.sum().over("s1_row").alias("g_ssum"),
        p.max().over("s1_row").alias("g_smax"),
        (p > 0.5).sum().over("s1_row").cast(pl.Float32).alias("g_s_strong"),
        (p > 0.5).sum().over("p_row").cast(pl.Float32).alias("g_p_strong"),
    )
    c = c.with_columns(
        # best competing S1 for this pool record (excluding this pair)
        pl.when(pl.col("g_rank_p") == 1)
          .then(p.top_k(2).over("p_row", mapping_strategy="join").list.get(1, null_on_oob=True))
          .otherwise(pl.col("g_pmax")).fill_null(0.0).alias("g_p_other"),
        # exclusivity-aware mass of the S1 entity
        pl.when(pl.col("g_rank_p") == 1).then(p).otherwise(0.0).alias("_pe"),
    )
    c = c.with_columns(pl.col("_pe").sum().over("s1_row").alias("g_ssum_excl"),
                       (pl.col("_pe") > 0.5).sum().over("s1_row").cast(pl.Float32).alias("g_s_strong_excl")).drop("_pe")
    c = c.with_columns((p - pl.col("g_p_other")).alias("g_p_margin"),
                       (pl.col("g_ssum") - p).alias("g_ssum_others"))
    return c


def sibling_features(store: Store, c: pl.DataFrame) -> pl.DataFrame:
    """For each pair (s1, p) compare p with the top-TOPQ other confident candidates q of s1."""
    top = (c.filter(pl.col("prob1") >= 0.3).sort(["s1_row", "prob1"], descending=[False, True])
             .group_by("s1_row", maintain_order=True).head(TOPQ + 1)
             .select("s1_row", pl.col("p_row").alias("q_row"), pl.col("prob1").alias("q_prob")))
    x = c.select("s1_row", "p_row").join(top, on="s1_row").filter(pl.col("p_row") != pl.col("q_row"))
    x = x.group_by("s1_row", "p_row", maintain_order=True).head(TOPQ)
    pr = x["p_row"].to_numpy().astype(np.int64)
    qr = x["q_row"].to_numpy().astype(np.int64)
    pool = store.pool
    feats = {}
    for nm, col, sc in [("sib_addr_ratio", "addr_full", fuzz.ratio), ("sib_addr_tset", "addr_full", fuzz.token_set_ratio),
                        ("sib_name_ratio", "name_core", fuzz.ratio), ("sib_name_tset", "name_core", fuzz.token_set_ratio)]:
        feats[nm] = process.cpdist(pool[col].gather(pr).to_list(), pool[col].gather(qr).to_list(),
                                   scorer=sc, workers=-1, dtype=np.float32)
    x = x.with_columns([pl.Series(k, v) for k, v in feats.items()])
    x = x.with_columns(
        (pl.col("sib_addr_ratio") * pl.col("q_prob")).alias("_wa"),
        (pl.col("sib_name_ratio") * pl.col("q_prob")).alias("_wn"),
        (pl.Series("sp", store.p_id.gather(pr).str.slice(0, 2)) == pl.Series("sq", store.p_id.gather(qr).str.slice(0, 2))).alias("_same_src"),
    )
    agg = x.group_by("s1_row", "p_row").agg(
        pl.col("sib_addr_ratio").max().alias("sib_addr_ratio_max"),
        pl.col("sib_addr_tset").max().alias("sib_addr_tset_max"),
        pl.col("sib_name_ratio").max().alias("sib_name_ratio_max"),
        pl.col("sib_name_tset").max().alias("sib_name_tset_max"),
        pl.col("_wa").max().alias("sib_addr_w"),
        pl.col("_wn").max().alias("sib_name_w"),
        pl.col("sib_addr_ratio").filter(pl.col("_same_src")).max().alias("sib_addr_same_src"),
        pl.col("sib_name_ratio").filter(pl.col("_same_src")).max().alias("sib_name_same_src"),
        pl.len().cast(pl.Float32).alias("sib_n"),
    )
    return c.join(agg, on=["s1_row", "p_row"], how="left").with_columns(pl.col("sib_n").fill_null(0.0))


def consistency_features(store: Store, c: pl.DataFrame) -> pl.DataFrame:
    """Noise hits records one at a time; a genuinely different business (decoy branch / sister
    company) repeats its difference across its own records. For each candidate, count how many
    OTHER candidates of the same S1 share its house number / exact name variant / address core."""
    pr = c["p_row"].to_numpy().astype(np.int64)
    sr = c["s1_row"].to_numpy().astype(np.int64)
    fn_p = store.num_first[pr + store.n1]
    fn_s = store.num_first[sr]
    c = c.with_columns(pl.Series("_fnp", fn_p), pl.Series("_fns", fn_s),
                       pl.Series("_np", store.pool["name_core"].gather(pr)),
                       pl.Series("_ns", store.s1["name_core"].gather(sr)),
                       pl.Series("_ap", store.pool["addr_core"].gather(pr)),
                       pl.Series("_src", store.p_id.gather(pr).str.slice(0, 2)))
    p = pl.col("prob1")
    # Source-aware support. The generator derives one noisy "base" record per source and copies it,
    # so SAME-source records of a true entity often share a wrong house number (65% of deviating
    # true records), while agreement ACROSS sources on a different number is rare for true entities
    # (14% of such groups) but common for planted decoy businesses (51%).
    has = pl.col("_fnp") >= 0
    n_num = pl.len().over("s1_row", "_fnp")
    n_num_src = pl.len().over("s1_row", "_src", "_fnp")
    m_num = p.sum().over("s1_row", "_fnp")
    m_num_src = p.sum().over("s1_row", "_src", "_fnp")
    c = c.with_columns(
        pl.when(has).then(n_num_src - 1).otherwise(-1).cast(pl.Float32).alias("cs_num_sup_same_src"),
        pl.when(has).then(n_num - n_num_src).otherwise(-1).cast(pl.Float32).alias("cs_num_sup_other_src"),
        pl.when(has).then(m_num_src - p).otherwise(-1.0).alias("cs_num_mass_same_src"),
        pl.when(has).then(m_num - m_num_src).otherwise(-1.0).alias("cs_num_mass_other_src"),
        # support for the S1's own house number, split by source relative to this record
        (pl.when(pl.col("_fnp") == pl.col("_fns")).then(p).otherwise(0.0).sum().over("s1_row", "_src")).alias("cs_s1num_mass_same_src"),
        (pl.when(pl.col("_fnp") == pl.col("_fns")).then(p).otherwise(0.0).sum().over("s1_row")
         - pl.when(pl.col("_fnp") == pl.col("_fns")).then(p).otherwise(0.0).sum().over("s1_row", "_src")).alias("cs_s1num_mass_other_src"),
        (pl.len().over("s1_row", "_src", "_np") - 1).cast(pl.Float32).alias("cs_name_sup_same_src"),
        (pl.len().over("s1_row", "_np") - pl.len().over("s1_row", "_src", "_np")).cast(pl.Float32).alias("cs_name_sup_other_src"),
    )
    c = c.with_columns(
        (pl.col("_fnp") == pl.col("_fns")).cast(pl.Float32).alias("cs_num_is_s1"),
        (pl.col("_np") == pl.col("_ns")).cast(pl.Float32).alias("cs_name_is_s1"),
        pl.when(pl.col("_fnp") >= 0).then(pl.len().over("s1_row", "_fnp") - 1).otherwise(-1).cast(pl.Float32).alias("cs_num_sup"),
        pl.when(pl.col("_fnp") >= 0).then(p.sum().over("s1_row", "_fnp") - p).otherwise(-1.0).alias("cs_num_mass"),
        (pl.len().over("s1_row", "_np") - 1).cast(pl.Float32).alias("cs_name_sup"),
        (p.sum().over("s1_row", "_np") - p).alias("cs_name_mass"),
        pl.when(pl.col("_ap") != "").then(pl.len().over("s1_row", "_ap") - 1).otherwise(-1).cast(pl.Float32).alias("cs_addr_sup"),
        pl.when(pl.col("_ap") != "").then(p.sum().over("s1_row", "_ap") - p).otherwise(-1.0).alias("cs_addr_mass"),
        # mass of candidates that carry the S1's own house number (the 'real' cluster)
        (pl.when(pl.col("_fnp") == pl.col("_fns")).then(p).otherwise(0.0).sum().over("s1_row")).alias("cs_s1num_mass"),
    )
    # SOURCE SLOTS. 92% of non-singleton businesses have >= 1 copy in each of Source 2 and Source 3, so an
    # S1 that has no confident record yet in this record's source is the likelier owner of an ambiguous
    # record (e.g. blank address, same name as another S1); compare with the best competing S1.
    strong = (p > 0.5).cast(pl.Int32)
    c = c.with_columns(
        (strong.sum().over("s1_row", "_src") - strong).cast(pl.Float32).alias("sl_same_src_strong"),
        (strong.sum().over("s1_row") - strong.sum().over("s1_row", "_src")).cast(pl.Float32).alias("sl_other_src_strong"),
        (p.sum().over("s1_row", "_src") - p).alias("sl_same_src_mass"),
    )
    top = (c.select("p_row", "s1_row", "prob1").sort(["p_row", "prob1"], descending=[False, True])
             .group_by("p_row", maintain_order=True).head(2)
             .with_columns(pl.int_range(pl.len()).over("p_row").alias("_k")))
    c = (c.join(top.filter(pl.col("_k") == 0).select("p_row", pl.col("s1_row").alias("_t1")), on="p_row", how="left")
          .join(top.filter(pl.col("_k") == 1).select("p_row", pl.col("s1_row").alias("_t2")), on="p_row", how="left"))
    c = c.with_columns(pl.when(pl.col("s1_row") == pl.col("_t1")).then(pl.col("_t2")).otherwise(pl.col("_t1")).alias("_comp"))
    slots = c.group_by("s1_row", "_src").agg(strong.sum().alias("_tc")).rename({"s1_row": "_comp"})
    c = c.join(slots, on=["_comp", "_src"], how="left")
    comp_p = pl.when(pl.col("s1_row") == pl.col("_t1")).then(pl.col("g_p_other")).otherwise(pl.col("g_pmax"))
    c = c.with_columns(
        pl.when(pl.col("_comp").is_null()).then(-1.0)
          .otherwise(pl.col("_tc").fill_null(0) - (comp_p > 0.5).cast(pl.Int32)).cast(pl.Float32).alias("sl_comp_same_src_strong"),
    ).drop("_t1", "_t2", "_comp", "_tc")
    # SIGNED house-number offset (pool - S1). Stage-1 number features are symmetric, but planted
    # branches sit a few doors UP (US train, same name: +1..2 -> 16% true, -1..2 -> 85% true).
    both = (pl.col("_fnp") >= 0) & (pl.col("_fns") >= 0)
    sd = (pl.col("_fnp") - pl.col("_fns")).cast(pl.Float32)
    c = c.with_columns(
        pl.when(both).then(sd.clip(-1000.0, 1000.0)).otherwise(None).alias("cs_num_sdiff"),
        pl.when(both & (pl.col("_fns") > 0)).then((sd / pl.col("_fns").cast(pl.Float32)).clip(-2.0, 2.0)).otherwise(None).alias("cs_num_srel"),
    )
    return c.drop("_fnp", "_fns", "_np", "_ns", "_ap", "_src")


def build_stage2(store: Store, c: pl.DataFrame) -> pl.DataFrame:
    # pairs below MIN_P1 carry negligible mass for sums / maxima -> drop them first (memory)
    keep = group_features(c.filter(pl.col("prob1") >= MIN_P1))
    keep = consistency_features(store, keep)
    return sibling_features(store, keep)
