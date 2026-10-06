#!/usr/bin/env bash
# resume after the rescore crash: stage-2 rescoring of the French-fixed test split (stage 1 reused),
# v13 submission file, then stage-2 ensemble partners + French-fixed test scoring.
SRC="<root>/code/business_entity_resolution/src"
ROOT="<root>"
cd "$SRC" || exit 1
PY="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
LOG=../../../work/v14.log
FILTER="deprecat|explode|^ *\)|StringCache|empty_as_null|UserWarning|warnings.warn|pl.concat\(\[c, f\]"
run() { echo "=== $* ($(date +%H:%M))" >> $LOG; "$@" 2>&1 | grep --line-buffered -v -E "$FILTER" >> $LOG; return ${PIPESTATUS[0]}; }
snap() {
  (cd "$ROOT" && $PY utils/validate_submission.py >> work/v14.log 2>&1)
  mkdir -p "<root>/submissions/$1" && cp <root>/output/*.tsv "<root>/submissions/$1/"
  echo "SNAP $1 ($(date +%H:%M))" >> $LOG
}
recal() {
  run $PY -c "
import polars as pl
from config import WORK
from recalibrate import recalibrate
recalibrate(pl.read_parquet(WORK + '/test_scored_$1.parquet'), pl.read_parquet(WORK + '/oof_$2.parquet', columns=['s1_row','p_row','prob'])).write_parquet(WORK + '/test_scored_$1r.parquet')
"
}
echo "--- v13b start ($(date +%H:%M))" >> $LOG
RESCORE_REUSE=1 run $PY rescore_test.py m12 x12 s12 --suffix f || exit 1
run $PY make_submission.py s12fr ownerF e12f || exit 1
snap v13_s12fr_france_fix
run $PY stage2_ens.py train e12 s12 || exit 1
run $PY stage2_ens.py test e12f s12f s12fens xb || exit 1
recal s12fens s12ens || exit 1
echo "V14_DONE ($(date +%H:%M))" >> $LOG
