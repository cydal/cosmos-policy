#!/usr/bin/env bash
# Pull Cosmos 3 weights into the HF cache on the ephemeral disk. Resumable; safe to
# re-run. Defaults to Cosmos3-Edge (4B) -- see config.py for why.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="/opt/dlami/nvme/cosmos-policy/.venv/bin/python"
REPO="${1:-nvidia/Cosmos3-Edge}"
DATA_DISK="/opt/dlami/nvme"

export HF_HOME="$DATA_DISK/cosmos-policy/hf"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=60

FREE_GB=$(df -BG --output=avail "$DATA_DISK" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 15 ]; then
  echo "Refusing to fetch: only ${FREE_GB}G free on $DATA_DISK." >&2
  exit 1
fi
echo "==> ${FREE_GB}G free on $DATA_DISK, fetching $REPO into HF_HOME=$HF_HOME"

"$PY" - "$REPO" <<'PY'
import sys, time
from huggingface_hub import snapshot_download

repo = sys.argv[1]
t0 = time.monotonic()
path = snapshot_download(repo_id=repo, max_workers=4)
print(f"{repo} -> {path}  ({(time.monotonic() - t0) / 60:.1f} min)")
PY
