#!/usr/bin/env bash
# Source this (don't execute it) before running anything against the venv:
#
#   source env.sh
#   python smoke_test.py
#
# Without it, loading the pipeline works but the VAE's conv3d dies on first use with
# CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED / "Unable to load any of
# {libcudnn_engines_runtime_compiled.so.9...}". The .so is right there in
# nvidia-cudnn-cu13's site-packages dir, but that dir is in nobody's default linker
# search path, so cudnn's own dlopen-by-soname lookup for its runtime-compiled JIT
# engine never finds it. Every nvidia-*-cu13 pip package ships its libs the same way,
# so this adds all of them, not just cudnn's.
VENV_DIR="/opt/dlami/nvme/cosmos-policy/.venv"
SITE="$VENV_DIR/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$(find "$SITE/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$VENV_DIR/bin:$PATH"
export HF_HOME="/opt/dlami/nvme/cosmos-policy/hf"
