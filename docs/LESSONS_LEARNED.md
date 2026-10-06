# Lessons learned: a post-competition review

After the final results we compared our pipeline, component by component, with a public top-50 solution:
[Paradox-73/Business_Entity_Resolution](https://github.com/Paradox-73/Business_Entity_Resolution), public LB 0.990565,
held-out 0.99226. Their numbers below come from their published README, experiment log and submission log.

## 1. Where the 0.0127 gap came from

| | Top-50 solution | Ours |
|---|---|---|
| Gradient-boosting pipeline only (stage 2 + expected-F0.5) | LB 0.9700 | **LB 0.9767** |
| Final public LB | 0.990565 | 0.977874 |
| US / India (estimated from probe uploads) | ≈ 0.992 | ≈ 0.983 |
| France (estimated from probe uploads) | ≈ 0.98 | ≈ 0.947 |
| Main hardware | laptop with RTX 3050 4 GB, plus 48 GB and 96 GB GPUs | laptop with RTX 3050 6 GB |

Our feature engineering and gradient-boosting stages were at least as strong as theirs. Roughly **0.007** of the gap is
US/India and **0.005** is France, and both came from components we did not build.

## 2. Component comparison

| Area | Top-50 solution | Ours | Their measured gain | Status |
|---|---|---|---|---|
| Final matcher for uncertain records | **fine-tuned cross-encoders** (e5-small on a 4 GB laptop GPU; bge-reranker-v2-m3, 568 M, on a 48 GB GPU) re-reading the 23.5 % "close calls"; 3 fold models; stage-3 XGBoost | frozen LaBSE embeddings as similarity features | +0.0070 held-out, ≈ +0.013 LB | missed |
| France decoy rules | lowercase-rate test (a generator quirk) showed French descriptor swaps ("Club" ↔ "Ecole") are look-alikes; 22 k pairs removed | probes on two other subsets | part of France 0.95 → 0.98 | missed |
| France legal-form veto | SARL vs SAS with no form in common → p = 0 | — | kept from their v7 on | missed |
| Transformer "min rule" for France | the transformer may only lower French probabilities | — (no transformer) | raising them cost −0.0035 France | missed |
| France noise words | "& Fils", "Services", "Groupe" accepted as noise; 9.8 k pairs added | found the same (v3) but did not turn it into additions | part of the France gain | partly |
| Blocking recall | audited at full density: name cap 4 k → 20 k, no-address k 100, dense e5 search → 99.63 % | `max_df` 3000 over S1 + pool → 98.5 % | +0.0032 held-out | missed |
| Cleaning audit | fixed `TN` → "tennessee" for India, ordinals | not audited | feature quality | missed |
| Group features (siblings, consensus, competition) | yes | yes, plus signed offsets and source slots | +0.0025 held-out | parity |
| Decision rule | expected-F0.5 per S1 | expected-F0.5 + one-owner renormalisation | — | parity |
| House-number recalibration for France | — | yes | — | ours only |
| Second pipeline + stacker | wider blocking on a 96 GB server, LightGBM stacker | single pipeline | +0.0005 LB | missed (hardware) |
| Validation protocol | held-out half searched against all train S1; held-out → LB gap ≈ 0.0017 | simulated test (trainsim); validation → LB gap ≈ 0.0084 | better decisions | weaker |

## 3. The three lessons

1. **Retrieval recall must be measured at full density, for every search channel.** A document-frequency cap that
   looks harmless on a sample can remove every token of a common name once the index is full size. Measure recall per
   channel, with the index as large as the test index.
2. **A cross-encoder beats similarity features.** A frozen bi-encoder gives one cosine number per pair. A cross-encoder
   reads both records token by token ("315 Laurel Circle" vs "31 Laurel Circle") and is trained on the labels. Applied only
   to uncertain records it is affordable even on a laptop GPU, and it was the largest single gain available.
3. **An unseen country needs label-free tests, not just probes.** Quirks of the data generator (for example, how often
   true vs fake records are all-lowercase) can estimate the true share of any group of pairs without labels. Probe uploads
   then confirm the rule. We tested only two subsets and missed the main French look-alike pattern.

## 4. What we would do with the same hardware

| Step | Expected gain | GPU needed |
|---|---|---|
| Raise the blocking caps, add a wider no-address search, rebuild | ≈ +0.001 – 0.003 | no |
| Fine-tune multilingual e5-small as a cross-encoder for close calls (frozen word table, 128 tokens, 1 epoch, 3 folds) + stage 3 | ≈ +0.006 – 0.007 | 6 GB laptop GPU |
| France rules from label-free tests (legal-form veto, descriptor veto, min rule), each confirmed by a France-only upload | ≈ +0.004 – 0.005 | no |
| Larger reranker (bge-reranker-v2-m3) and a second pipeline | ≈ +0.001 | 24 GB+ GPU |
