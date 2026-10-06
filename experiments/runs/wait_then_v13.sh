#!/usr/bin/env bash
W="<root>/work"
until grep -q "S12R_DONE" "$W/v12.log"; do
  if grep -q "Traceback" "$W/v12.log"; then echo "v12 failed - v13 not started ($(date +%H:%M))" >> "$W/v13.log"; exit 1; fi
  sleep 30
done
bash "$W/run_v13.sh"
