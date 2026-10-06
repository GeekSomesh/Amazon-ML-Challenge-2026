# Business Entity Resolution — Team Zephyrus

Amazon ML Challenge 2026. For every Source-1 business record, find all records in Source 2 and Source 3 that refer to the
same real-world business (metric: macro F0.5 per Source-1 entity).

**Final leaderboard score: 0.977874** · validation macro F0.5: 0.98623

The methodology, analysis and results are described in [`docs/METHODOLOGY.md`](../../docs/METHODOLOGY.md) and the [main README](../../README.md).
This folder contains everything needed to regenerate both output files from the provided data.

---

## 1. Pipeline overview

```
 raw TSV ─► normalisation ─► blocking (sparse IDF top-k, both directions) ─► pairwise features
        ─► stage 1: LightGBM + XGBoost  (also acts as a learned second blocking stage, ~5 candidates per S1)
        ─► LaBSE multilingual embeddings of names and addresses
        ─► stage 2: LightGBM re-scorer  (competition, sibling, consistency and LaBSE similarity features)
        ─► decision layer (unseen-country recalibration, one owner per record, expected-F0.5 selection)
        ─► output/matching_results.tsv  +  output/candidate_pairs.tsv
```

| Stage | What it does |
|---|---|
| Normalisation | Accent folding, Indic-script → Latin dictionary learned from training pairs, OCR-digit repair, legal forms / honorifics / aliases, address abbreviations, state tables, removal of injected address tokens (found unsupervised by comparing Source-1 and Source-2/3 token frequencies per country). |
| Training simulation | `trainsim` = all training Source-1 entities + synthetic planted look-alikes that reproduce the statistical signature of the test pool (≈1.1 per entity). Used for training and validation. |
| Blocking | Per country, binary IDF-weighted sparse vectors (name/address unigrams + bigrams, name character 4-grams, name×house-number keys); top-20 Source-1→pool and top-6 pool→Source-1 cosine neighbours. ≈42 candidates per Source-1 entity, pair recall 98.5 %. |
| Stage 1 | ~71 string / token / number / context features; LightGBM + GPU XGBoost, 2-fold cross-fitting by Source-1 entity. Pairs with probability ≥ 0.005 go to stage 2 (≈5 per entity) — this set is `candidate_pairs.tsv`. |
| Stage 2 | Group features (competition between owners, Source-1 group mass), sibling agreement, per-source consistency of house-number deviations, signed house-number offset, per-source slot counts, and LaBSE cosine / rank / gap features for names and addresses. |
| Decision | Label-free house-number noise-rate recalibration for countries absent from training (France); one owner per pool record; per Source-1 entity, the candidate prefix that maximises expected F0.5. |

---

## 2. Requirements

* Python 3.11; packages pinned in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```
  `torch` must be the CUDA build that matches the local driver (tested: 2.12.0 + CUDA 13.2).
* NVIDIA GPU with ≥ 6 GB memory (XGBoost `device="cuda"` and LaBSE encoding), 16 GB RAM, ~150 GB free disk for
  intermediate files.
* Pretrained model: `sentence-transformers/LaBSE` (Apache-2.0, 471 M parameters) is downloaded automatically by
  `sentence-transformers` on first use. No other external data, API or service is used.
* Reference machine: Windows 11, 16 logical cores, 16 GB RAM, RTX 3050 Laptop GPU (6 GB). Full run ≈ 8 hours.

## 3. Data layout and paths

```
<root>/
├── dataset/
│   ├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
│   └── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
├── code/business_entity_resolution/      (this folder)
├── work/                                 intermediate files (created automatically)
└── output/                               submission files (created automatically)
```

`<root>` is resolved as three levels above `src/`. Override with the environment variables `BER_DATA`, `BER_WORK`,
`BER_OUTPUT` (and `BER_THREADS` for the CPU thread count).

## 4. Run

```bash
bash run_pipeline.sh
```

The script runs the steps below in order and ends with the official format validator.

| # | Command (from `src/`) | Output (in `work/` unless noted) |
|---|---|---|
| 1 | `python convert.py` | parquet copies of all TSVs, `train_gt.parquet` |
| 2 | `python learn_translit.py` | `translit.json` (Indic → Latin dictionary) |
| 3 | `python prepare.py train test` | `{split}_s{1,2,3}_norm.parquet` |
| 4 | `BER_KEEP_FRAC=1.0 python make_sim.py` | `trainsim_s1_norm.parquet`, `trainsim_gt.parquet` |
| 5 | `BER_DECOY_RATE=0.85 python make_decoys.py` | `trainsim_s{2,3}_norm.parquet` (pool + planted look-alikes) |
| 6 | `python blocking.py {trainsim,test} --k_fwd 20 --k_rev 6 --max_df 3000 --char_w 0.5 --combo_w 1.0` | `{split}_cand.parquet` |
| 7 | `python build_features.py trainsim trainsim_cand.parquet` / `test test_cand.parquet` | `feat_{split}/part_*.parquet` |
| 8 | `python train_lgb.py m12` | LightGBM fold models, `oof_m12.parquet` |
| 9 | `python predict_test.py m12 ownerF` | `test_scored_m12.parquet` |
| 10 | `BER_XGB_FRAC=0.22 python train_xgb.py x12 m12` | XGBoost fold models, `oof_e12` / `test_scored_e12` (LGB + XGB average) |
| 11 | `python emb_encode.py` · `python emb_encode.py addr_full` | LaBSE vectors of names / addresses |
| 12 | `python emb_pairs.py trainsim` · `EMB_KIND=addr python emb_pairs.py trainsim` · `python emb_sib.py trainsim` | LaBSE pair and sibling features |
| 13 | `python stage2_emb.py train e12` | final stage-2 models, `oof_s12a.parquet`, validation F0.5 |
| 14 | the three commands of step 12 with `test e12` | LaBSE features for test pairs |
| 15 | `python stage2_emb.py test e12 s12a` | `test_scored_s12a.parquet` |
| 16 | `python recalibrate.py s12a` | `test_scored_s12ar.parquet` |
| 17 | `python make_submission.py s12ar ownerF e12` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

`python eval_oof.py <tag>` prints the validation macro F0.5 of any out-of-fold file.

## 5. Outputs

* `output/matching_results.tsv` — one row per test Source-1 entity, comma-separated matched Source-2/3 ids (empty for
  predicted singletons).
* `output/candidate_pairs.tsv` — exactly the pairs scored by the final stage-2 model (stage-1 probability ≥ 0.005);
  every predicted match is a member of this set.

The leaderboard files are not versioned in this repository (≈230 MB). A re-run
reproduces them up to minor numerical differences (multi-threaded tree training, GPU floating point and tie-breaking in
nearest-neighbour retrieval).

## 6. Results

| Model | Validation macro F0.5 | Leaderboard |
|---|---|---|
| Stage 1 (LightGBM + XGBoost) | 0.97814 | — |
| Stage 2 | 0.98575 | 0.976662 |
| Stage 2 + LaBSE name features | 0.98619 | 0.977756 |
| **Stage 2 + LaBSE name and address features (final)** | **0.98623** | **0.977874** |

Validation: 2-fold cross-fitting by Source-1 entity on the test-faithful `trainsim` split (US and India).

## 7. Models and licences

| Component | Licence | Size |
|---|---|---|
| LightGBM | MIT | gradient-boosted trees |
| XGBoost | Apache-2.0 | gradient-boosted trees |
| LaBSE (`sentence-transformers/LaBSE`) | Apache-2.0 | 471 M parameters |

All models are within the challenge rules (MIT / Apache-2.0, ≤ 8 B parameters). Only the provided training and test
data are used.

## 8. Source code

`src/` — production pipeline

| Module | Role |
|---|---|
| `config.py` | paths, thread count, split-aware file lookup |
| `convert.py` | raw TSV → parquet |
| `learn_translit.py` | Indic-script → Latin token dictionary and native-script state names from training pairs |
| `text_norm.py` | name and address normalisation |
| `prepare.py` | applies normalisation; removes injected address tokens (unsupervised, per country) |
| `make_sim.py`, `make_decoys.py` | test-faithful training / validation split with synthetic planted look-alikes |
| `blocking.py` | candidate generation (sparse IDF vectors, `sparse_dot_topn`, both directions) |
| `features.py`, `build_features.py` | pairwise string / token / number / context features (rapidfuzz, numba) |
| `train_lgb.py`, `train_xgb.py`, `predict_test.py` | stage-1 models and test scoring |
| `stage2.py`, `train_stage2.py` | stage-2 frame: group, sibling, consistency, signed-offset and slot features |
| `emb_encode.py`, `emb_complete.py`, `emb_pairs.py`, `emb_sib.py` | LaBSE embeddings and embedding features |
| `stage2_emb.py` | final stage-2 model (training, validation, test scoring) |
| `recalibrate.py` | label-free house-number noise-rate recalibration for countries unseen in training |
| `decide.py`, `metric.py`, `postfilter.py` | one-owner renormalisation, expected-F0.5 selection, macro F0.5, decoy-word filter |
| `eval_oof.py` | validation report |
| `make_submission.py`, `write_outputs.py` | submission files |
| `package_submission.py` | builds the submission archive |

`src/analysis/` — diagnostics and experiments behind the documented design decisions (loss decomposition, adversarial
validation, ablations of decision rules, leaderboard probe builders). They are not needed to reproduce the output.
