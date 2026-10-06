# Experiment journal

This journal records every version we built, the score it got and the decision we took. Unless stated otherwise, "validation"
means macro F0.5 on out-of-fold predictions (2-fold cross-fitting by Source-1 entity) with the owner + expected-F0.5
decision rule. Raw logs are in [`../experiments/logs/`](../experiments/logs/) and the orchestration scripts are in
[`../experiments/runs/`](../experiments/runs/).

---

## 1. Validation setups

The biggest methodological lesson of the competition was that **the validation set must be built the way the test set was
built**. We used three setups over time:

| Setup | Built how | Pool per S1 | Used for |
|---|---|---|---|
| A: `trainsim` v1 | 18.7 % of training S1 entities removed *before* blocking. Their records stay in the pool as realistic orphans | 5.754 | v1 – v5 |
| B: `trainsim` v1 + decoys | setup A + synthetic planted look-alikes (25 % of entities) | > 6 | v6 – v10 |
| **C: `trainsim` final** | **all** training S1 entities + 2.42 M synthetic planted look-alikes that match the test signature | 5.77 | v12 → final |

Why the final change: removing S1 entities makes the pool-per-S1 ratio right, but it also makes the Source-1 index *less dense*
than the real one, which distorts every candidate-list feature (counts, ranks, margins). It also does not create the kind of
look-alike the test set contains (same name, house number a few doors away). Keeping every S1 entity and planting
signature-matched decoys fixes both. Adversarial validation on setup C reaches only AUC 0.565 for the US. Scores from
different setups are not comparable: setup A is the easiest, setup C the hardest and the closest to test.

---

## 2. Version timeline

| Version | Change | Validation | Public LB | Decision |
|---|---|---|---|---|
| v1 | First normalisation, two-directional sparse blocking, 61 pairwise features, LightGBM stage 1 | 0.9781 (A) | **0.970443** | baseline |
| v2 | + stage-2 re-scorer: competition and sibling-agreement features | 0.9803 (A) | **0.973196** | keep |
| v3 | v2 + removal of French "groupe / développement" words treated as decoy markers | — | **0.972680** | revert: these words are generator noise |
| v4 | Blocking upgrades (name character 4-grams: recall 0.9801 → 0.9872), normalisation and feature upgrades | 0.9850 (A) | **0.974386** | keep |
| v5 | + label-free house-number recalibration for France | — | **0.974815** | keep |
| v6 | First decoy-aware stage 1: on decoy-heavy data 0.9775 vs 0.9503 for v5's model | 0.9775 (B) | not uploaded | folded into v7 |
| v7 | Decoy-trained LightGBM + XGBoost ensemble, stage 2 | 0.98405 (B) | **0.975781** | keep |
| probe | v7 with every French prediction emptied | — | **0.842796** | France ≈ 0.94, US/India ≈ 0.98 |
| v9 | Source-aware consistency features (same-source vs other-source support) | — | not uploaded | carried into v12 |
| v10 | v9 + French recall additions (base-mates, acronyms) | — | **0.975499** | revert |
| **v12** | **Setup C** (all S1 + planted decoys), signed house-number offset and source-slot features, stage 1 as learned second blocker (≈5 candidates per S1) | **0.98575 (C)** | **0.976662** | keep |
| v13 | v12 stage 2 applied to a France-renormalised test build | — | not uploaded | tested via v14 |
| v14 | 3-model stage-2 ensemble (+0.00006 validation) + France-only normalisation of injected words | 0.98581 | **0.975909** | revert France normalisation |
| P1 | v12 minus French pairs whose house number differs | — | **0.976050** | those pairs were correct |
| P2 | v12 minus French pairs with an added marker word | — | **0.976007** | those pairs were correct |
| s12m | v12 + LaBSE name embeddings (cosine, rank, gap) | 0.98613 | not uploaded | keep |
| s12n | s12m + LaBSE sibling similarity | 0.98619 | **0.977756** | keep. LB gain 2.5× validation gain |
| **s12a** | **s12n + LaBSE address embeddings (final)** | **0.98623** | **0.977874** | **final submission** |

Fourteen leaderboard submissions in total (v1, v2, v3, v4, v5, v7, probe, v10, v12, v14, P1, P2, s12n, s12a).

---

## 3. What the leaderboard taught us about France

France is 15 % of the test entities and has no labels, so French changes could only be measured on the leaderboard. Every
France experiment was uploaded as a file in which **only French rows differ** from a reference upload, so the score change
divided by 0.15 is the change in French F0.5.

| Upload | Δ LB vs reference | Δ France F0.5 | Reading |
|---|---|---|---|
| v3 (drop "groupe / développement" records) | −0.00052 vs v2 | ≈ −0.0034 | these words are noise on true copies |
| v5 (house-number recalibration) | +0.00043 vs v4 | ≈ +0.0029 | France really has less house-number noise |
| v10 (recall additions; also carried v9's US/India changes, so not a clean France test) | −0.00028 vs v7 | — | no gain |
| P1 (drop number-changed pairs) | −0.00061 vs v12 | ≈ −0.0041 | the kept number-changed pairs are true |
| P2 (drop marker-word pairs) | −0.00066 vs v12 | ≈ −0.0044 | the kept marker-word pairs are true |
| s12n (LaBSE) | +0.00109 vs v12 | — (mixed) | validation gain on US/India was only +0.00044, so most of the gain is French |

The emptied-France probe puts French F0.5 at ≈ 0.94 vs ≈ 0.98 for US/India, which is the largest single gap in the
submission. The language-agnostic LaBSE features were the only change that clearly closed part of it.

---

## 4. Decision-rule ablation (stage 1 `m12`, trainsim final)

| Rule | Macro F0.5 |
|---|---|
| threshold 0.5, no constraint | 0.97461 |
| exclusivity (record → best S1) + threshold 0.5 | 0.97594 |
| exclusivity + threshold 0.7 (best threshold) | 0.97752 |
| exclusivity + expected-F0.5 | 0.97750 |
| one-owner renormalisation + threshold 0.6 | 0.97774 |
| **one-owner + expected-F0.5** | **0.97787** |

The same ordering held at stage 2 (`s12`: 0.98443 → 0.98575).

---

## 5. Stage-2 variants (trainsim final)

| Tag | Model | Features | Validation | Notes |
|---|---|---|---|---|
| s12 | LightGBM | 73 (group, sibling, consistency, offset, slots) | 0.98575 | log-loss 0.0437 / 0.0450 per fold |
| s12ens | LightGBM + XGBoost + second LightGBM | 73 | 0.98581 | +0.00006, 3× the cost |
| s13 | stage 3 on top of s12 | 72 | 0.98574 | no gain |
| s12m | LightGBM + LaBSE names | 77 | 0.98613 | most useful LaBSE features: `emb_gap_p`, `emb_cos`, `emb_rank_s1` |
| s12n | + LaBSE sibling similarity | 80 | 0.98619 | `emb_sib_same_src` ranks 5th of 80 |
| **s12a** | **+ LaBSE addresses** | **84** | **0.98623** | log-loss 0.0415 / 0.0427 (−5 % vs s12) |

---

## 6. Blocking experiments

Measured on `trainsim` v1, India (the harder country), against the ground truth:

| Variant | Recall | Candidates per S1 |
|---|---|---|
| word unigrams + bigrams, `max_df` 3000, top-20 / top-6 | 0.9646 | 41.9 |
| + name character 4-grams (weight 0.5) | **0.9706** | 42.1 |
| same, pruned to S1-rank ≤ 15 or record-rank ≤ 4 | 0.9654 | 28.8 |
| same, pruned to S1-rank ≤ 10 or record-rank ≤ 3 | 0.9592 | 20.4 |
| same, pruned to S1-rank ≤ 5 or record-rank ≤ 2 | 0.9471 | 12.2 |

Pruning the retrieval lists cost too much recall, so we kept all ≈42 candidates and used the stage-1 model as a *learned*
second blocker (p ≥ 0.005 → ≈5 per S1) instead. On the final setup, recall is 0.9852 (US 0.9869, India 0.9827).

---

## 7. Ideas rejected after validation

| Idea | Result |
|---|---|
| Hand-written boost / penalty rules (heavy-noise, number-up, number-down) | −0.0001 … −0.0009 |
| One house-number cluster per source per entity | −0.003 |
| Identical-name sibling rule | precision 42 % |
| "Sole occupant" acceptance | ~150 k extra false positives on validation |
| Per-entity empty / count model | ±0 |
| Conservative agreement variants | no gain |
| France-only filler / name / address canonicalisation | −0.00075 LB |

---

## 8. Engineering notes

* **16 GB RAM is the binding constraint.** The first blocking implementation needed 19 GB and paged. It was rewritten to
  build the sparse matrices chunk by chunk under a global string cache and to load only the needed columns. Only one heavy
  job runs at a time.
* **GPU memory:** XGBoost on the 6 GB laptop GPU is trained on a 22 % sample of the pairs per fold. LaBSE encodes in fp16
  with sequence length 32 (names) / 48 (addresses). Embeddings are reduced to 128 dims with PCA and stored as float16
  (12.85 M addresses ≈ 3.3 GB).
* **Caching:** the stage-2 training frame is cached as parquet (`cached_frame`), and embeddings are keyed by the string,
  so rebuilding test only encodes strings it has not seen (`emb_complete.py`).
* **Logging:** long runs are chained in bash scripts with filtered, line-buffered logs (`experiments/runs/`), so a
  failure stops the chain at the failing step.
* **Reproducibility:** the refactored code base was verified to reproduce the final leaderboard files byte for byte from
  the saved intermediate files.
