#!/usr/bin/env bash
# after v12 (S12R_DONE): v12 submission file, stage 3, France-only test re-normalisation + re-scoring.
SRC="<root>/code/business_entity_resolution/src"
ROOT="<root>"
cd "$SRC" || exit 1
PY="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
LOG=../../../work/v13.log
FILTER="deprecat|explode|^ *\)|StringCache|empty_as_null|UserWarning|warnings.warn|pl.concat\(\[c, f\]"
run() { echo "=== $* ($(date +%H:%M))" >> $LOG; "$@" 2>&1 | grep --line-buffered -v -E "$FILTER" >> $LOG; return ${PIPESTATUS[0]}; }
snap() {  # snap <name>: validate output/ and copy it to <root>\submissions\<name>
  (cd "$ROOT" && $PY utils/validate_submission.py >> work/v13.log 2>&1)
  mkdir -p "<root>/submissions/$1" && cp <root>/output/*.tsv "<root>/submissions/$1/"
  echo "SNAP $1 ($(date +%H:%M))" >> $LOG
}
# 1) v12 submission (stage-2 + France recalibration; candidates = pairs reaching stage 2)
run $PY make_submission.py s12r ownerF e12 || exit 1
snap v12_s12r_candidates_reduced
# 2) stage 3 = stage-2 builder fed with stage-2 OOF probabilities
run $PY train_stage2.py s12 s13 || exit 1
run $PY eval_oof.py s13 || exit 1
# 3) France-only re-normalisation of the test inputs (US / India rows are unchanged), rebuild, re-score
run $PY prepare.py test || exit 1
run $PY blocking.py test --k_fwd 20 --k_rev 6 --max_df 3000 --char_w 0.5 --combo_w 1.0 || exit 1
run $PY build_features.py test test_cand.parquet || exit 1
run $PY rescore_test.py m12 x12 s12 s13 --suffix f || exit 1
echo "V13_DONE ($(date +%H:%M))" >> $LOG
