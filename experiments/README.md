# Experiment records

Raw records of the runs made during the competition. They complement [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).

| Folder | Content |
|---|---|
| `runs/` | the bash scripts that chained the long runs (one script per version or experiment) |
| `logs/` | their logs: blocking recall, feature build timings, per-fold log-loss, validation tables, embedding throughput, submission checks |
| `feature_lists/` | the exact feature lists of the trained models (`m12` = stage 1, `s12m` / `s12n` / `s12a` = stage 2 with LaBSE features) |

Notes:

* The scripts are kept as they were run. Local paths are replaced by `<root>`, and the interpreter by `$PYTHON` / `python`.
  Some scripts call experiment modules that now live in `code/business_entity_resolution/src/analysis/`
  (for example `rescore_test.py` and `stage2_ens.py`).
* The maintained end-to-end entry point is
  [`code/business_entity_resolution/run_pipeline.sh`](../code/business_entity_resolution/run_pipeline.sh).

| Script → log | What it ran |
|---|---|
| `run_v12.sh` → `v12.log` | final `trainsim` rebuild: blocking, features, stage 1 (LightGBM + XGBoost), stage 2, recalibration |
| `run_v13.sh`, `run_v13b.sh` → `v13.log` | v12 submission, stage-3 experiment, France-renormalised test build |
| `run_v14.sh`, `run_v14b.sh` → `v14.log` | 3-model stage-2 ensemble, France-only normalisation test |
| `run_v15.sh` → `v15.log` | test rebuilt without the France-only rules |
| `run_emb*.sh` → `emb*.log` | LaBSE name embeddings, pair features, sibling features (s12m, s12n) |
| `run_addr.sh` → `addr.log` | LaBSE address embeddings and the final model s12a |
