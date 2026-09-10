#!/usr/bin/env bash
# Generate a bigger, more diverse policy-training dataset, train with a wall-clock
# safety cap (not a fixed epoch count) and best-val-accuracy checkpointing, then
# closed-loop-evaluate on all three splits -- chained, and fully OS-detached
# (setsid+nohup+disown, same pattern as run_lora_scaleup.sh) so it survives this
# session/connection going away for a few hours.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="/opt/dlami/nvme/cosmos-policy/policy_scaleup"
DATASET_ROOT="$RUN_DIR/dataset"
CKPT_OUT="$RUN_DIR/checkpoints"
LOG="$RUN_DIR/run.log"
PIDFILE="$RUN_DIR/run.pid"

MUJOCO_PY="/opt/dlami/nvme/mujoco-env-dataset-venv/bin/python"
POLICY_PY="/opt/dlami/nvme/cosmos-policy/.venv/bin/python"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running: PID $(cat "$PIDFILE")" >&2
  exit 1
fi

mkdir -p "$CKPT_OUT"
source "$HERE/env.sh"

setsid nohup bash -c "
set -euo pipefail
echo '=== [1/3] generating dataset ($(date)) ==='
cd '$HERE/../mujoco-env-dataset'
export MUJOCO_GL=egl
'$MUJOCO_PY' scripts/generate_dataset.py \
  --out '$DATASET_ROOT' \
  --train '${TRAIN_EPISODES:-2000}' --val '${VAL_EPISODES:-150}' --test '${TEST_EPISODES:-150}' \
  --image-size 256 --seed '${SEED:-0}'

echo '=== [2/3] training ($(date)) ==='
cd '$HERE'
'$POLICY_PY' train_policy.py \
  --dataset-root '$DATASET_ROOT' \
  --out '$CKPT_OUT' \
  --epochs '${EPOCHS:-40}' \
  --max-hours '${MAX_HOURS:-2.0}' \
  --batch-size 128 --num-workers 3

echo '=== [3/3] closed-loop eval ($(date)) ==='
for split in train val test; do
  echo \"--- \$split ---\"
  '$POLICY_PY' eval_policy_closed_loop.py \
    --dataset-root '$DATASET_ROOT' \
    --checkpoint '$CKPT_OUT/best.pt' \
    --split \"\$split\" --n-episodes '${EVAL_EPISODES:-50}' --max-steps 160
done
echo '=== done ($(date)) ==='
" > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
disown

echo "launched PID $(cat "$PIDFILE")"
echo "log:     $LOG"
echo "pid:     $PIDFILE"
echo "dataset: $DATASET_ROOT"
echo "ckpt:    $CKPT_OUT"
