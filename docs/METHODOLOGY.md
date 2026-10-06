# Methodology — Amazon ML Challenge 2026: Business Entity Resolution

**Team Name:** Zephyrus
**Team Members:** Somesh Srivastava, Naina Agarwal, Paraj Prasad, Harshitha Jagannath — Bangalore Institute of Technology (BIT), Bangalore
**Submission Date:** 27 Sep 2026

---

## 1. Executive Summary
A precision-first entity-resolution pipeline: data-driven normalisation (Indic→Latin dictionary learned from training
pairs, OCR-digit repair, unsupervised removal of injected tokens), two-directional sparse IDF top-k blocking, a two-stage
gradient-boosted matcher (LightGBM + XGBoost ensemble, then a stage-2 re-scorer with competition, sibling-agreement and
source-aware consistency features) and a one-owner / expected-F0.5 decision layer. Two ideas carried the result:
(1) a **training set rebuilt to mirror the test set's construction** — unlabelled statistics show that ~20% of the test pool
are planted look-alike records (≈1.15 per Source-1 entity), so signature-matched decoys were synthesised for training; and
(2) **language-agnostic LaBSE embeddings** of names and addresses as stage-2 features, which let what is learned on US/India
transfer to the unseen French data (+0.0011 on the leaderboard for +0.0004 on US/India validation).
The stage-1 matcher doubles as a learned blocker, so the final model scores only ~5 candidates per Source-1 entity.
**Final leaderboard score: 0.977874** (validation macro F0.5 0.98623).

---

## 2. Methodology

### 2.1 Problem Analysis
* **Scale:** train S1 2.21M, S2 5.03M, S3 5.29M; test S1 1.73M (India 0.81M, US 0.66M, France 0.26M), S2 4.89M, S3 5.08M.
* **Labels:** 5.6% singletons; 3.46 matches per S1 (1.67 S2 + 1.79 S3, max 11); no Source-2/3 record in two lists
  (exclusivity); country always equal inside a true pair (country-partitioned processing; country treated as an open set).
* **Names are ambiguous:** ~50% of S1 records share their normalised name with another S1 record → the address and in
  particular the house number decide.
* **Scripts:** ~20% of Indian S2/S3 names are word-by-word transliterations into 9 Indic scripts (token counts agree
  99.999% with the English name) → exact dictionary learned from training pairs; native address fragments = state names.
* **Name noise:** typos, OCR digits (exactly 0→o 1→l 5→s 8→b 6→g), accents, case, token duplication, word drops (~28%),
  legal-form drops/changes (~40%), token splits, domains, acronyms, alias constructions with generated random names,
  honorifics; **address noise:** component reordering, abbreviations, state full/abbr/native, county for city, digit drops
  and hyphen injection in numbers, ordinal noise, PO Box/PMB, null tokens, 3.3% blank addresses.
* **Generator structure:** one noisy *base* record per source per business, copied with light per-record noise: in 91.8% of
  (business, source) groups with ≥2 true records all copies carry the same house number; a deviating number is shared
  within the source 65% of the time but across sources only 12% (decoy groups: 51% cross-source).
* **Test ≠ train (key finding):** pool/S1 is 5.75 in test vs 4.68 in train. Blank-address, domain-name and alias rates in the
  test pool are all **0.80×** the training rates while cosmetic noise (Indic script, null tokens, caps) is unchanged → ~20% of
  the test pool are planted records that never receive heavy per-record noise, i.e. ≈1.15 look-alike decoys per S1
  ("branch"/"sister" businesses: same name, house number a few doors away, sometimes a branch/holding word).
* **France (test only, no labels):** among confident French matches the house number differs in only ~0.9% of pairs
  (US 13%, India 6%), and French pool names carry injected words (Participations, Holding, Distribution, International,
  Groupe, Développement ≈34k–54k each, almost absent from Source 1) that also mark planted look-alikes.

### 2.2 Solution Strategy
**Approach Type:** Hybrid — learned normalisation, sparse retrieval, two-stage gradient boosting, constraint-aware decisions.
**Core Innovation:** a test-faithful training/validation set (all S1 kept + signature-matched synthetic planted decoys),
source-aware "consistency of deviations" features, multilingual (LaBSE) name/address similarity for cross-language
generalisation, label-free noise-rate recalibration for the unseen country, and an expected-F0.5 decision layer under a
one-owner constraint.

---

## 3. Candidate Generation (Blocking)
* **Vectors (per country):** binary IDF-weighted sparse vectors over name-core unigrams+bigrams, address unigrams+bigrams
  (house numbers included), name character 4-grams (w=0.5) and composite name-token×house-number keys (w=1.0); tokens seen
  once or in >3,000 records are dropped.
* **Retrieval:** `sparse_dot_topn` top-20 S1→pool and top-6 pool→S1 cosine neighbours, union (≈42 pairs/S1).
  Recall on training pairs: 98.5–98.7% depending on the decoy density (India 98.3–98.5%, US 98.7–98.9%); oracle
  macro-F0.5 ceiling ≈0.996.
* **Learned second blocking stage:** the stage-1 LightGBM+XGBoost ensemble keeps pairs with probability ≥ 0.005 → **≈5
  candidates per S1** (`candidate_pairs.tsv` = exactly the pairs the final stage-2 model scores; every match is among them).
* **Why nothing is lost:** both retrieval directions (a pool record always meets its best S1 candidates), composite keys for
  truncated addresses, and recall measured against the ground truth after every change.

---

## 4. Matching Model
**Stage-1 features (≈71):** name — rapidfuzz ratio/token-sort/token-set/partial/Jaro-Winkler on the core name, glued name
(domains/splits), glued full name incl. legal form, IDF-weighted overlaps, missing IDF mass, legal-form agreement,
acronym match, share of never-seen tokens (generated names), decoy-word indicator, name commonness; address — rapidfuzz on
full and "core" address (per-country frequent tokens removed), IDF overlaps, house-number equality / presence / digit edit
distance / numeric difference / digit-length difference, blank-address flag, address commonness; context — blocking
cosine, ranks in both lists, gap and margin to the best competitor.
**Stage-2 features:** stage-1 probability, competition (best other S1 for the record, margin, mass), S1 group mass,
sibling agreement with the S1's most confident candidates (overall / same source), consistency counts and masses for the
record's house number, exact name variant and address core — split into same-source vs other-source support.
**Multilingual similarity (LaBSE, Apache-2.0, 471M params):** every normalised name (9.8M) and address (12.9M) occurring in a
stage-2 pair is embedded with LaBSE (language-agnostic sentence encoder, fp16 on one RTX 3050; PCA to 128 dims). Stage 2
receives, for names and for addresses, the cosine, its rank / gap among the S1's candidates and among the pool record's
competing owners, plus (names) the similarity to the S1's other confident candidates overall and in the same source (copies
of one business share a per-source base record). These features are language-independent, so what is learned on US / India
pairs transfers to Indic-script names and to the unseen French data; they cut stage-2 log-loss by ~5%.
**Model type:** LightGBM (MIT) + XGBoost (Apache-2.0, GPU) ensemble for stage 1, LightGBM for stage 2 (with LaBSE features);
2-fold cross-fitting by S1 entity so every pair has an out-of-fold probability.
**Decision layer:** one-owner renormalisation q = o/(1+Σo) per pool record, keep its best owner, then per S1 the prefix of
candidates maximising expected F0.5 (vs. predicting nothing); training-derived decoy-word filter; for countries unseen in
training, label-free house-number noise-rate recalibration (odds of number-mismatch pairs × r_country / r_train, France ≈ 0.09).

---

## 5. Results & Error Analysis
* **Leaderboard history:**
  - v1 0.9704 (stage 1) → v2 0.9732 (stage 2) → v4 0.9744 (blocking / normalisation / features) → v5 0.9748 (France
    recalibration) → v7 0.9758 (decoy-aware LGB + XGB)
  - v12 0.976662: test-faithful simulation, signed-offset + source-slot stage-2 features, learned second blocking stage
  - + LaBSE name features 0.977756 (validation only +0.0004 but leaderboard +0.0011 → the language-agnostic features mostly
    helped the unseen French data)
  - **+ LaBSE address features 0.977874 (final)**
  - France-only probes: France-specific normalisation of injected words lowered the score (0.9759); removing selected French
    pairs with a different house number / a marker word lowered it as well (0.9761 / 0.9760) → those selections are correct;
    the France-only rules were therefore switched off.
* **Validation** (test-faithful split: all 2.21M training S1 entities, ≈1.1 planted look-alikes per entity; full table in
  Appendix B): stage 1 0.97814 → stage 2 0.98575 → **final 0.98623** (India 0.98500, US 0.98705). Adversarial validation
  (a classifier separating test from validation candidate pairs on content features) reaches only AUC 0.565 for the US,
  i.e. the simulation matches the test distribution well.
* **Rejected after validation (kept out of the submission):** hand-written boost/penalty rules (−0.0001…−0.0009),
  one-house-number-cluster-per-source filter (−0.003), stage 3 (±0.0000), identical-name sibling rule (42% precision).
* **Loss decomposition (validation, 2.2M S1):** stage 1 misses 331k true pairs vs 33k false merges (house-number noise 155k,
  blank address 132k). After stage 2: 172k misses (blank address 115k = 40% of blank-address true pairs; house-number noise
  29k; generated names 16k) + 113k true pairs never retrieved, vs 18k false merges — the remaining loss is mostly genuinely
  ambiguous blank-address copies of businesses that share their name with other Source-1 entities.
* **Common false positives:** planted look-alike businesses at a nearby house number, co-located businesses with a swapped
  category word, single-record look-alikes of singleton S1 entities.
* **Common false negatives:** copies with blank or heavily truncated addresses and very common names; random replacement
  names with truncated addresses; house numbers changed by per-source noise.

---

## 6. Conclusion
Understanding how the data was generated mattered more than model capacity: reproducing the test set's hidden composition
in training, and reasoning about groups of records (competition, per-source bases, cross-source agreement), produced the
largest gains, while every normalisation rule was learned or validated from the data itself. For the country that never
appears in training, language-agnostic multilingual embeddings were the most effective lever: they improved the leaderboard
score roughly 2.5× more than the US/India validation gain, i.e. they transferred what was learned to French records.
Hand-crafted rules, by contrast, were consistently rejected by out-of-fold validation or by controlled leaderboard probes.

---

## Appendix
### A. Code Artefacts
`code/business_entity_resolution/run_pipeline.sh` runs the complete pipeline (raw TSV → both output files) and ends with
the official format validator; `README.md` documents every step, its command and its output. Production modules live in
`src/` (normalisation `text_norm.py` / `prepare.py`, training simulation `make_sim.py` / `make_decoys.py`, blocking
`blocking.py`, features `features.py` / `build_features.py`, stage 1 `train_lgb.py` / `train_xgb.py`, stage 2 `stage2.py` /
`train_stage2.py`, LaBSE features `emb_encode.py` / `emb_pairs.py` / `emb_sib.py`, final model `stage2_emb.py`, decision
layer `recalibrate.py` / `decide.py` / `make_submission.py`). `src/analysis/` holds the diagnostics and experiments behind
the design decisions reported here (loss decomposition, adversarial validation, rule ablations, leaderboard probes).

### B. Additional Results
**Validation (test-faithful split, owner + expected-F0.5 decision), macro F0.5:**

| Model | India | US | All |
|---|---|---|---|
| stage 1 LightGBM | 0.97544 | 0.97949 | 0.97787 |
| stage 1 LightGBM + XGBoost | — | — | 0.97814 |
| stage 2 | 0.98440 | 0.98665 | 0.98575 |
| stage 2 + LaBSE names + siblings | 0.98494 | 0.98702 | 0.98619 |
| stage 2 + LaBSE names + siblings + addresses (final) | 0.98500 | 0.98705 | 0.98623 |

**Leaderboard probes used to understand the unseen country (France):**

| Submission | Leaderboard | Reading |
|---|---|---|
| v7 with every French prediction emptied | 0.8428 | France F0.5 minus French singleton rate = 0.888 |
| v12 | 0.976662 | baseline for the France probes |
| v12 − French pairs whose house number differs | 0.976050 | those pairs were correct |
| v12 − French pairs with added marker words | 0.976007 | those pairs were correct |
| France-only normalisation of injected words | 0.975909 | hurt -> switched off |
| + LaBSE name features | 0.977756 | +0.0011 on LB vs +0.0004 on validation: the gain is mostly French |
| + LaBSE address features (final) | **0.977874** | |

**Validated and rejected (not in the final model):** hand-written boost/penalty rules, one-number-cluster-per-source filter
(−0.003), stage 3 (±0), per-S1 empty/count model (±0), conservative agreement variants, identical-name sibling rule.

### C. Reproducibility and Compliance
* **Data:** only the provided training and test files are used; no external databases, APIs, geocoding or look-ups.
* **Models:** LightGBM (MIT), XGBoost (Apache-2.0), LaBSE `sentence-transformers/LaBSE` (Apache-2.0, 471 M parameters) —
  all within the MIT / Apache-2.0, ≤ 8 B parameter rule.
* **Hardware / runtime:** 16 logical cores, 16 GB RAM, NVIDIA RTX 3050 (6 GB); full pipeline ≈ 8 hours, ≈ 150 GB of
  intermediate files. The `output/` files in this archive are the exact leaderboard files; a re-run reproduces them up to
  minor numerical differences (multi-threaded tree training, GPU floating point, nearest-neighbour tie-breaking).
