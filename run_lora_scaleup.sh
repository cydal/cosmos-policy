#!/usr/bin/env bash
# Launch the scaled-up LoRA pilot as a fully detached background process: immune to
# this shell, this SSH/VSCode session, or the Claude Code harness dying or losing
# connection (e.g. the laptop driving this session going to sleep). Three layers,
# each defends against a different way a "background" process can still die:
#
#   setsid    new session, no controlling terminal -- a closed terminal can't
#             deliver SIGHUP/SIGTERM to a process with no controlling tty at all.
#   nohup     ignore SIGHUP anyway, belt and braces.
#   disown    removes it from this shell's job table, so this shell exiting
#             doesn't send it anything either.
#
# Monitor with ./monitor_lora_scaleup.sh -- do not attach to this process directly;
# it has no controlling terminal to attach to by design.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="/opt/dlami/nvme/cosmos-policy/lora_scaleup"
DATASET_ROOT="$RUN_DIR/dataset"
CKPT_OUT="$RUN_DIR/checkpoints"
LOG="$RUN_DIR/train.log"
PIDFILE="$RUN_DIR/train.pid"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running: PID $(cat "$PIDFILE")" >&2
  exit 1
fi

mkdir -p "$CKPT_OUT"
source "$HERE/env.sh"
PY="/opt/dlami/nvme/cosmos-policy/.venv/bin/python"

cd "$HERE"
setsid nohup "$PY" train_lora_pilot.py \
  --steps "${STEPS:-24000}" \
  --max-hours "${MAX_HOURS:-10.75}" \
  --grad-accum 4 \
  --lr 1e-4 \
  --stride "${STRIDE:-4}" \
  --ckpt-every 500 \
  --dataset-root "$DATASET_ROOT" \
  --out "$CKPT_OUT" \
  > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
disown

echo "launched PID $(cat "$PIDFILE")"
echo "log:  $LOG"
echo "pid:  $PIDFILE"
echo "ckpt: $CKPT_OUT"
