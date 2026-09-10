#!/usr/bin/env bash
# Launch the Cosmos v2 fine-tune (exposure-bias-aware training, COSMOS_FINETUNE_V2.md)
# fully OS-detached, same pattern as run_lora_scaleup.sh: setsid+nohup+disown, so it
# survives this session/connection going away for the full ~18h run. Given the tight
# 20h event deadline this is demoing in, this run MUST NOT be lost to a dropped
# connection -- verify detachment (PPID==1, no controlling tty) immediately after
# launch, same as every previous long run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="/opt/dlami/nvme/cosmos-policy/cosmos_v2"
CKPT_OUT="$RUN_DIR/checkpoints"
LOG="$RUN_DIR/run.log"
PIDFILE="$RUN_DIR/run.pid"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running: PID $(cat "$PIDFILE")" >&2
  exit 1
fi

mkdir -p "$CKPT_OUT"
source "$HERE/env.sh"
PY="/opt/dlami/nvme/cosmos-policy/.venv/bin/python"

cd "$HERE"
setsid nohup "$PY" train_cosmos_v2.py \
  --max-hours "${MAX_HOURS:-18}" \
  --grad-accum 4 \
  --lr 1e-4 \
  --stride 4 \
  --ckpt-every 500 \
  --photoreal-prob "${PHOTOREAL_PROB:-0.4}" \
  --self-rollout-every "${SELF_ROLLOUT_EVERY:-100}" \
  --self-rollout-steps "${SELF_ROLLOUT_STEPS:-20}" \
  --out "$CKPT_OUT" \
  > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
disown

echo "launched PID $(cat "$PIDFILE")"
echo "log:  $LOG"
echo "pid:  $PIDFILE"
echo "ckpt: $CKPT_OUT"
