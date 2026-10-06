#!/usr/bin/env bash
# after EMB_DONE (LaBSE embeddings) and S12GR_DONE (test rebuilt without France rules):
# pair features -> stage 2 with LaBSE features (validation) -> test scoring -> France recalibration
W="<root>/work"
until grep -q "EMB_DONE" "$W/emb.log" && grep -q "S12GR_DONE" "$W/v15.log"; do
  grep -q "Traceback" "$W/emb.log" "$W/v15.log" && { echo "upstream failed" >> "$W/emb2.log"; exit 1; }; sleep 30; done
cd "$(dirname "$0")/../../code/business_entity_resolution/src" || exit 1
PY="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
LOG=../../../work/emb2.log
FILTER="deprecat|explode|^ *\)|StringCache|empty_as_null|UserWarning|warnings.warn|pl.concat\(\[c, f\]|Loading weights|Warn"
run() { echo "=== $* ($(date +%H:%M))" >> $LOG; "$@" 2>&1 | grep --line-buffered -v -E "$FILTER" >> $LOG; return ${PIPESTATUS[0]}; }
run $PY emb_pairs.py trainsim || exit 1
run $PY stage2_emb.py train || exit 1
run $PY emb_pairs.py test e12g || exit 1
run $PY stage2_emb.py test e12g s12mg || exit 1
run $PY -c "
import polars as pl
from config import WORK
from recalibrate import recalibrate
recalibrate(pl.read_parquet(WORK + '/test_scored_s12mg.parquet'), pl.read_parquet(WORK + '/oof_s12m.parquet', columns=['s1_row','p_row','prob'])).write_parquet(WORK + '/test_scored_s12mgr.parquet')
" || exit 1
echo "EMB2_DONE ($(date +%H:%M))" >> $LOG
