#!/usr/bin/env bash
# Health check for the detached policy scale-up run (generate -> train -> eval).
set -euo pipefail

RUN_DIR="/opt/dlami/nvme/cosmos-policy/policy_scaleup"
PIDFILE="$RUN_DIR/run.pid"
LOG="$RUN_DIR/run.log"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  PID="$(cat "$PIDFILE")"
  echo "== running: PID $PID =="
  ps -o pid,etime,pcpu,pmem,cmd -p "$PID" 2>/dev/null || pgrep -a -P "$PID" || true
else
  echo "== NOT running (pidfile missing or stale) =="
fi

echo
echo "== last log lines =="
tail -n 8 "$LOG" 2>/dev/null || echo "(no log yet)"

echo
echo "== GPU =="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

echo
echo "== checkpoints =="
ls -1t "$RUN_DIR/checkpoints" 2>/dev/null | head -5

echo
echo "== disk (ephemeral) =="
df -h /opt/dlami/nvme | tail -1
