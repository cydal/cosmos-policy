#!/usr/bin/env bash
# Set up NVIDIA Cosmos 3 on this box: x86_64, one discrete L40S (46 GiB VRAM),
# CUDA 13.2 driver. Idempotent; safe to re-run.
#
# The venv lives on /opt/dlami/nvme (217 GiB free ephemeral instance storage), not
# the root disk, which currently has ~11 GiB free. Nothing here is precious: if the
# ephemeral disk is wiped across a stop/start, just re-run this script.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="/opt/dlami/nvme/cosmos-policy/.venv"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

echo "==> venv (python3.12, on ephemeral disk)"
mkdir -p "$(dirname "$VENV_DIR")"
if [ ! -d "$VENV_DIR" ]; then
  python3.12 -m venv "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"
"$PY" -m pip install -q --upgrade pip

# torch FIRST and from the cu130 index, same reasoning as the DGX Spark setup: a
# later resolve can otherwise pull a CPU-only wheel from plain PyPI over the top,
# and the only symptom is torch.cuda.is_available() == False much later.
echo "==> torch (cu130, x86_64)"
"$PY" -m pip install -q --index-url "$TORCH_INDEX" torch torchvision

echo "==> cosmos stack"
"$PY" -m pip install -q -r "$HERE/requirements.txt"

echo "==> verify"
source "$HERE/env.sh"
"$PY" - <<'PY'
import torch
print(f"  torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"  device: {torch.cuda.get_device_name(0)}  free {free/2**30:.1f} / {total/2**30:.1f} GiB")
    print(f"  capability: sm_{''.join(str(c) for c in torch.cuda.get_device_capability(0))}")

import diffusers, transformers
print(f"  diffusers {diffusers.__version__}  transformers {transformers.__version__}")

from diffusers import Cosmos3OmniPipeline, CosmosActionCondition  # noqa: F401
print("  Cosmos3OmniPipeline imports")
import av  # noqa: F401
print("  PyAV imports")
PY

cat <<EOF

Done. Next:

  ./fetch.sh                        # pull nvidia/Cosmos3-Edge (~few GB) once
  source env.sh && python smoke_test.py   # one short clip on the host
EOF
