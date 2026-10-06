#!/usr/bin/env bash
# after EMB2_DONE: LaBSE sibling-similarity features -> stage 2 (s12n) validation -> test scoring
W="<root>/work"
until grep -q "EMB2_DONE" "$W/emb2.log"; do grep -q "Traceback" "$W/emb2.log" && { echo "emb2 failed" >> "$W/emb3.log"; exit 1; }; sleep 30; done
cd "$(dirname "$0")/../../code/business_entity_resolution/src" || exit 1
PY="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
LOG=../../../work/emb3.log
FILTER="deprecat|explode|^ *\)|StringCache|empty_as_null|UserWarning|warnings.warn|pl.concat\(\[c, f\]|Loading weights|Warn"
run() { echo "=== $* ($(date +%H:%M))" >> $LOG; "$@" 2>&1 | grep --line-buffered -v -E "$FILTER" >> $LOG; return ${PIPESTATUS[0]}; }
run $PY emb_sib.py trainsim || exit 1
EMB_SIB=1 run $PY stage2_emb.py train || exit 1
run $PY emb_sib.py test e12g || exit 1
EMB_SIB=1 run $PY stage2_emb.py test e12g s12ng || exit 1
run $PY -c "
import polars as pl
from config import WORK
from recalibrate import recalibrate
recalibrate(pl.read_parquet(WORK + '/test_scored_s12ng.parquet'), pl.read_parquet(WORK + '/oof_s12n.parquet', columns=['s1_row','p_row','prob'])).write_parquet(WORK + '/test_scored_s12ngr.parquet')
" || exit 1
echo "EMB3_DONE ($(date +%H:%M))" >> $LOG
