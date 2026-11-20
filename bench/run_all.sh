#!/usr/bin/env bash
# Runs all benchmark phases while logging GPU power once per second.
set -euo pipefail
cd "$(dirname "$0")"
POOL=${POOL:-308058}
nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,clocks.sm --format=csv,noheader,nounits -lms 1000 > power.csv &
PW=$!
trap 'kill $PW 2>/dev/null' EXIT
: > phases.txt
for part in prefill decode context; do
  echo "$part start $(date +%s)" >> phases.txt
  python3 -I bench.py --only $part --pool-tokens $POOL --out results-$part.json
  echo "$part end $(date +%s)" >> phases.txt
done
