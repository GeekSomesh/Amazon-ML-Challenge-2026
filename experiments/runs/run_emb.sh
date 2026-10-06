#!/usr/bin/env bash
cd "$(dirname "$0")/../../code/business_entity_resolution/src" || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
python emb_encode.py 2>&1 | grep --line-buffered -v -E "Warn|warn|Loading weights" >> ../../../work/emb.log
