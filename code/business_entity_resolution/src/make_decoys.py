"""Inject synthetic HARD decoys into the 'trainsim' pool.

Analysis of the unlabelled test pool showed that planted test decoys ('different business planted next to a real one')
copy the real business's name and address almost exactly and differ mainly in the house
number (offset distribution measured on test marker-word decoys: same number 14%, +-1..2 16%,
+-3..10 35%, +-11..100 21%, >100 2%, digit-count change 12%), while training decoys differ
more in the name - so the model learned to reject decoys by name and accepts test decoys.
Here we copy genuine Source-2/3 records of random training entities, move them to another house
number (and/or add a sister-company word), and add them to the pool as NON-matches, so the model
learns that "same name, house number a few doors away" is a different business.
Output: work/trainsim_s2_norm.parquet, work/trainsim_s3_norm.parquet (train pool + decoys).
"""
import os

import numpy as np
import polars as pl

from config import WORK

RATE = float(os.environ.get("BER_DECOY_RATE", "0.25"))  # share of S1 entities that receive a decoy entity
N_REC = [1, 2, 3]          # records per decoy entity
N_REC_P = [0.5, 0.35, 0.15]
MARKERS = {"US": ["holdings", "westgate", "riverside", "southside", "midtown", "lakeside", "northside",
                  "eastgate", "greater", "central", "uptown", "downtown"],
           "India": ["holdings", "infratech", "overseas"]}


def new_number(n, rng):
    """offset drawn from the test decoy offset distribution (excluding 'same number')"""
    u = rng.random()
    s = str(n)
    if u < 0.186:                             # +-1..2  (16/86)
        d = rng.integers(1, 3)
    elif u < 0.186 + 0.407:                   # +-3..10 (35/86)
        d = rng.integers(3, 11)
    elif u < 0.593 + 0.244:                   # +-11..100 (21/86)
        d = rng.integers(11, 101)
    elif u < 0.837 + 0.023:                   # >100 (2/86)
        d = rng.integers(101, 1000)
    else:                                     # digit-count change (12/86)
        return int(s + str(rng.integers(0, 10))) if (rng.random() < 0.5 or len(s) == 1) else int(s[:-1])
    m = n + d if (rng.random() < 0.5 or n - d < 1) else n - d
    if len(str(m)) != len(s) and n - d >= 1:
        m = n - d if m == n + d else n + d
    return int(m)


def main(seed=7):
    rng = np.random.default_rng(seed)
    s1 = pl.read_parquet(os.path.join(WORK, "trainsim_s1_norm.parquet"), columns=["entity_id", "country", "addr_full"])
    s1 = s1.with_columns(pl.col("addr_full").str.extract(r"\b(\d+)\b").alias("s1num"))
    pools = {src: pl.read_parquet(os.path.join(WORK, f"train_s{src}_norm.parquet")) for src in (2, 3)}
    pool = pl.concat([pools[2], pools[3]])
    gt = (pl.read_parquet(os.path.join(WORK, "trainsim_gt.parquet"))
            .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1_id", "matched_entity_ids": "entity_id"}))
    # candidate donor records: true records that contain the S1 house number as a token
    d = (gt.join(pool, on="entity_id").join(s1.rename({"entity_id": "s1_id", "country": "c1", "addr_full": "a1"}), on="s1_id")
           .filter(pl.col("s1num").is_not_null())
           # planted test decoys never carry the heavy per-record noise (domain names, aliases, blank
           # address): in test these rates are 0.80x the training rates, exactly the 20% dilution
           .filter(~pl.col("is_domain") & ~pl.col("has_alias"))
           .filter(pl.col("addr_full").str.contains(pl.lit(r"\b") + pl.col("s1num") + pl.lit(r"\b"))))
    # deterministic: fixed row order before every random draw
    d = d.sort(["s1_id", "entity_id"])
    ents = d.select("s1_id").unique().sort("s1_id").sample(fraction=RATE, seed=seed)
    d = d.join(ents, on="s1_id", maintain_order="left").with_columns(pl.int_range(pl.len()).shuffle(seed=seed).over("s1_id").alias("_r"))
    k = pl.DataFrame({"s1_id": ents["s1_id"], "_k": rng.choice(N_REC, size=ents.height, p=N_REC_P)})
    d = d.join(k, on="s1_id", maintain_order="left").filter(pl.col("_r") < pl.col("_k")).sort(["s1_id", "entity_id"])
    # per decoy entity: one new number (or same number + marker word), shared by all its records
    ent = d.select("s1_id", "s1num", "country").unique("s1_id", maintain_order=True)
    same = rng.random(ent.height) < 0.14
    newn = [s if sm else str(new_number(int(s[:12]), rng)) for s, sm in zip(ent["s1num"].to_list(), same)]
    add_marker = same | (rng.random(ent.height) < 0.30)
    mk = [rng.choice(MARKERS.get(c, MARKERS["US"])) if a else "" for c, a in zip(ent["country"].to_list(), add_marker)]
    ent = ent.with_columns(pl.Series("newnum", newn), pl.Series("mk", mk))
    d = d.join(ent.select("s1_id", "newnum", "mk"), on="s1_id", maintain_order="left")
    def swap(addr, old, new):
        toks = addr.split(" ")
        for i, t in enumerate(toks):
            if t == old:
                toks[i] = new
                break
        return " ".join(toks)

    d = d.with_columns(pl.Series("addr_full", [swap(a, o, n) for a, o, n in zip(d["addr_full"].to_list(), d["s1num"].to_list(), d["newnum"].to_list())]))
    d = d.with_columns(
        pl.when(pl.col("mk") != "").then(pl.col("name_full") + pl.lit(" ") + pl.col("mk")).otherwise(pl.col("name_full")).alias("name_full"),
        pl.when(pl.col("mk") != "").then(pl.col("name_core") + pl.lit(" ") + pl.col("mk")).otherwise(pl.col("name_core")).alias("name_core"),
    ).with_columns(
        pl.col("name_core").str.replace_all(" ", "").alias("name_join"),
        pl.col("name_core").str.split(" ").list.sort().list.join(" ").alias("name_sorted"),
        (pl.col("entity_id").str.slice(0, 3) + pl.lit("DCY") + pl.int_range(pl.len()).cast(pl.String)).alias("entity_id"),
    )
    cols = pools[2].columns
    dec = d.select(cols)
    for src in (2, 3):
        part = dec.filter(pl.col("entity_id").str.starts_with(f"S{src}-"))
        pl.concat([pools[src], part]).write_parquet(os.path.join(WORK, f"trainsim_s{src}_norm.parquet"))
        print(f"S{src}: +{part.height} decoy records")
    print(f"decoy entities {ent.height} ({ent.height / s1.height:.3f} per S1), records {dec.height}; "
          f"same-number+marker {int(same.sum())}, with marker {int(add_marker.sum())}")


if __name__ == "__main__":
    main()
