#!/usr/bin/env bash
# French-fixed test: stage-2 re-score (reusing the stage-1 ensemble scores) + stage-2 ensemble partners.
cd "$(dirname "$0")/../../code/business_entity_resolution/src" || exit 1
PY="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
LOG=../../../work/v14.log
FILTER="deprecat|explode|^ *\)|StringCache|empty_as_null|UserWarning|warnings.warn|pl.concat\(\[c, f\]"
run() { echo "=== $* ($(date +%H:%M))" >> $LOG; "$@" 2>&1 | grep --line-buffered -v -E "$FILTER" >> $LOG; return ${PIPESTATUS[0]}; }
RESCORE_REUSE=1 run $PY rescore_test.py m12 x12 s12 --suffix f || exit 1
echo "S12FR_DONE ($(date +%H:%M))" >> $LOG
run $PY stage2_ens.py train e12 s12 || exit 1
run $PY stage2_ens.py test e12f s12f s12fens xb || exit 1
run $PY -c "
import polars as pl
from config import WORK
from recalibrate import recalibrate
recalibrate(pl.read_parquet(WORK + '/test_scored_s12fens.parquet'), pl.read_parquet(WORK + '/oof_s12ens.parquet', columns=['s1_row','p_row','prob'])).write_parquet(WORK + '/test_scored_s12fensr.parquet')
" || exit 1
echo "V14_DONE ($(date +%H:%M))" >> $LOG
