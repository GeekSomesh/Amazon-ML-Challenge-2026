#!/usr/bin/env bash
# after EMB3_DONE: persist missing test name embeddings, recompute LaBSE test features, re-score s12n and s12m
W="<root>/work"
until grep -q "EMB3_DONE" "$W/emb3.log"; do grep -q "Traceback" "$W/emb3.log" && { echo "emb3 failed" >> "$W/emb4.log"; exit 1; }; sleep 20; done
cd "$(dirname "$0")/../../code/business_entity_resolution/src" || exit 1
PY="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
LOG=../../../work/emb4.log
FILTER="deprecat|explode|^ *\)|StringCache|empty_as_null|UserWarning|warnings.warn|pl.concat\(\[c, f\]|Loading weights|Warn"
run() { echo "=== $* ($(date +%H:%M))" >> $LOG; "$@" 2>&1 | grep --line-buffered -v -E "$FILTER" >> $LOG; return ${PIPESTATUS[0]}; }
recal() { run $PY -c "
import polars as pl
from config import WORK
from recalibrate import recalibrate
recalibrate(pl.read_parquet(WORK + '/test_scored_$1.parquet'), pl.read_parquet(WORK + '/oof_$2.parquet', columns=['s1_row','p_row','prob'])).write_parquet(WORK + '/test_scored_$1r.parquet')
"; }
run $PY emb_complete.py test e12g || exit 1
run $PY emb_pairs.py test e12g || exit 1
run $PY emb_sib.py test e12g || exit 1
EMB_SIB=1 run $PY stage2_emb.py test e12g s12ng || exit 1
recal s12ng s12n || exit 1
run $PY stage2_emb.py test e12g s12mg || exit 1
recal s12mg s12m || exit 1
echo "EMB4_DONE ($(date +%H:%M))" >> $LOG
