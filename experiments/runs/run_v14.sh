#!/usr/bin/env bash
# after v13: stage-2 ensemble partners (XGB + second LGB), French-fixed test scoring, recalibration.
W="<root>/work"
until grep -q "V13_DONE" "$W/v13.log"; do grep -q "Traceback" "$W/v13.log" && { echo "v13 failed - v14 not started" >> "$W/v14.log"; exit 1; }; sleep 30; done
cd "$(dirname "$0")/../../code/business_entity_resolution/src" || exit 1
PY="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
LOG=../../../work/v14.log
FILTER="deprecat|explode|^ *\)|StringCache|empty_as_null|UserWarning|warnings.warn|pl.concat\(\[c, f\]"
run() { echo "=== $* ($(date +%H:%M))" >> $LOG; "$@" 2>&1 | grep --line-buffered -v -E "$FILTER" >> $LOG; return ${PIPESTATUS[0]}; }
recal() {  # recal <scored_tag> <oof_tag> -> test_scored_<scored_tag>r
  run $PY -c "
import polars as pl
from config import WORK
from recalibrate import recalibrate
recalibrate(pl.read_parquet(WORK + '/test_scored_$1.parquet'), pl.read_parquet(WORK + '/oof_$2.parquet', columns=['s1_row','p_row','prob'])).write_parquet(WORK + '/test_scored_$1r.parquet')
"
}
recal s12f s12 || exit 1
run $PY stage2_ens.py train e12 s12 || exit 1
run $PY stage2_ens.py test e12f s12f s12fens xb || exit 1
recal s12fens s12ens || exit 1
echo "V14_DONE ($(date +%H:%M))" >> $LOG
