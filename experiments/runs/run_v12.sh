#!/usr/bin/env bash
# v12: training = all S1 + signature-matched planted decoys (already written); rebuild from blocking.
cd "$(dirname "$0")/../../code/business_entity_resolution/src" || exit 1
PY="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8
LOG=../../../work/v12.log
FILTER="deprecat|explode|^ *\)|StringCache|empty_as_null|UserWarning|warnings.warn|pl.concat\(\[c, f\]"
run() { echo "=== $* ($(date +%H:%M))" >> $LOG; "$@" 2>&1 | grep --line-buffered -v -E "$FILTER" >> $LOG; return ${PIPESTATUS[0]}; }
export PYTHONUNBUFFERED=1
run $PY blocking.py trainsim --k_fwd 20 --k_rev 6 --max_df 3000 --char_w 0.5 --combo_w 1.0 || exit 1
run $PY build_features.py trainsim trainsim_cand.parquet || exit 1
run $PY train_lgb.py m12 || exit 1
run $PY eval_oof.py m12 || exit 1
run $PY build_features.py test test_cand.parquet || exit 1
run $PY predict_test.py m12 ownerF || exit 1
BER_XGB_FRAC=0.22 run $PY train_xgb.py x12 m12 || exit 1
run $PY eval_oof.py e12 || exit 1
run $PY train_stage2.py e12 s12 || exit 1
run $PY eval_oof.py s12 || exit 1
run $PY -c "
import polars as pl
from config import WORK
from recalibrate import recalibrate
recalibrate(pl.read_parquet(WORK + '/test_scored_s12.parquet'), pl.read_parquet(WORK + '/oof_s12.parquet', columns=['s1_row','p_row','prob'])).write_parquet(WORK + '/test_scored_s12r.parquet')
" || exit 1
echo "S12R_DONE ($(date +%H:%M))" >> $LOG
