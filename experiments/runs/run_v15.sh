#!/usr/bin/env bash
# test rebuilt WITHOUT the France-only normalisation (LB showed it hurt) -> v12-equivalent inputs, suffix g
cd "$(dirname "$0")/../../code/business_entity_resolution/src" || exit 1
PY="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
LOG=../../../work/v15.log
FILTER="deprecat|explode|^ *\)|StringCache|empty_as_null|UserWarning|warnings.warn|pl.concat\(\[c, f\]"
run() { echo "=== $* ($(date +%H:%M))" >> $LOG; "$@" 2>&1 | grep --line-buffered -v -E "$FILTER" >> $LOG; return ${PIPESTATUS[0]}; }
run $PY prepare.py test || exit 1
run $PY blocking.py test --k_fwd 20 --k_rev 6 --max_df 3000 --char_w 0.5 --combo_w 1.0 || exit 1
run $PY build_features.py test test_cand.parquet || exit 1
run $PY rescore_test.py m12 x12 s12 --suffix g || exit 1
echo "S12GR_DONE ($(date +%H:%M))" >> $LOG
