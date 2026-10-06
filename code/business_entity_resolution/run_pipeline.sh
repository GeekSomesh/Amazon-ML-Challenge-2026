#!/usr/bin/env bash
# =====================================================================================================
#  Business Entity Resolution - end-to-end pipeline (team Zephyrus)
#
#  raw TSVs (dataset/)  ->  output/matching_results.tsv  +  output/candidate_pairs.tsv
#
#  Usage (from this folder):   bash run_pipeline.sh
#  Paths (optional overrides): BER_DATA   (default <root>/dataset)
#                              BER_WORK   (default <root>/work)     intermediate files, ~150 GB
#                              BER_OUTPUT (default <root>/output)
#  <root> is the folder three levels above src/ (see src/config.py). PYTHON selects the interpreter.
#  Requirements: Python 3.11, packages in requirements.txt, an NVIDIA GPU (XGBoost, LaBSE), 16 GB RAM.
#  Runtime on the reference machine (16 cores, 16 GB RAM, RTX 3050 6 GB): about 8 hours.
# =====================================================================================================
set -euo pipefail
cd "$(dirname "$0")/src"
PY="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
step() { echo; echo "==> [$(date +%H:%M:%S)] $*"; "$@"; }

# ---- 1. data preparation ----------------------------------------------------------------------------
step "$PY" convert.py                         # TSV -> parquet (+ training ground truth)
step "$PY" learn_translit.py                  # Indic-script -> Latin token dictionary from training pairs
step "$PY" prepare.py train test              # normalisation + unsupervised removal of injected address tokens

# ---- 2. test-faithful training / validation set -----------------------------------------------------
step env BER_KEEP_FRAC=1.0 "$PY" make_sim.py      # keep every training Source-1 entity
step env BER_DECOY_RATE=0.85 "$PY" make_decoys.py # add planted look-alikes matching the test pool signature

# ---- 3. candidate generation (blocking) -------------------------------------------------------------
for split in trainsim test; do
  step "$PY" blocking.py "$split" --k_fwd 20 --k_rev 6 --max_df 3000 --char_w 0.5 --combo_w 1.0
done

# ---- 4. pairwise features ---------------------------------------------------------------------------
step "$PY" build_features.py trainsim trainsim_cand.parquet
step "$PY" build_features.py test test_cand.parquet

# ---- 5. stage 1: LightGBM + XGBoost (2-fold S1-entity cross-fitting) -------------------------------
step "$PY" train_lgb.py m12                         # out-of-fold probabilities oof_m12
step "$PY" predict_test.py m12 ownerF               # test scores test_scored_m12
step env BER_XGB_FRAC=0.22 "$PY" train_xgb.py x12 m12   # XGBoost (GPU) + LGB/XGB average -> oof_e12, test_scored_e12
step "$PY" eval_oof.py e12                          # validation report of the stage-1 ensemble

# ---- 6. multilingual (LaBSE) embeddings of every name / address in a stage-2 pair (GPU) --------------
step "$PY" emb_encode.py                            # names      (builds the cached stage-2 training frame first)
step "$PY" emb_encode.py addr_full                  # addresses
step "$PY" emb_complete.py test e12                 # embed any remaining test string (no-op on a clean run)
step env EMB_KIND=addr "$PY" emb_complete.py test e12

# ---- 7. final stage 2 (group / sibling / consistency features + LaBSE features) ---------------------
step "$PY" emb_pairs.py trainsim
step env EMB_KIND=addr "$PY" emb_pairs.py trainsim
step "$PY" emb_sib.py trainsim
step "$PY" stage2_emb.py train e12                  # models lgb_s12a_fold*, oof_s12a, validation F0.5
step "$PY" emb_pairs.py test e12
step env EMB_KIND=addr "$PY" emb_pairs.py test e12
step "$PY" emb_sib.py test e12
step "$PY" stage2_emb.py test e12 s12a              # test_scored_s12a

# ---- 8. decision layer and submission files --------------------------------------------------------
step "$PY" recalibrate.py s12a                      # house-number noise-rate recalibration for unseen countries
step "$PY" make_submission.py s12ar ownerF e12      # one-owner + expected-F0.5 selection -> output/*.tsv

# ---- 9. format check with the official validator ----------------------------------------------------
ROOT="$(cd ../../.. && pwd)"
if [ -f "$ROOT/utils/validate_submission.py" ]; then
  step "$PY" "$ROOT/utils/validate_submission.py" --matching "${BER_OUTPUT:-$ROOT/output}/matching_results.tsv" \
       --candidate "${BER_OUTPUT:-$ROOT/output}/candidate_pairs.tsv" --test-dir "${BER_DATA:-$ROOT/dataset}/test"
fi
echo; echo "Pipeline finished."
