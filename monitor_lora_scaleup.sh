#!/usr/bin/env bash
# Quick health check for the detached overnight run: is it alive, how far along,
# what's the recent loss trend, how's GPU/disk. Safe to run repeatedly.
set -euo pipefail

RUN_DIR="/opt/dlami/nvme/cosmos-policy/lora_scaleup"
PIDFILE="$RUN_DIR/train.pid"
LOG="$RUN_DIR/train.log"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  PID="$(cat "$PIDFILE")"
  echo "== running: PID $PID =="
  ps -o pid,etime,pcpu,pmem,cmd -p "$PID"
else
  echo "== NOT running (pidfile missing or stale) =="
fi

echo
echo "== last log lines =="
tail -n 6 "$LOG" 2>/dev/null || echo "(no log yet)"

echo
echo "== GPU =="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

echo
echo "== checkpoints =="
ls -1 "$RUN_DIR/checkpoints" 2>/dev/null | tail -5

echo
echo "== disk (ephemeral) =="
df -h /opt/dlami/nvme | tail -1
