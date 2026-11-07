#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# Keep machine settings (Python, GPUs, model locations) out of the repository.
if [[ -f .env.local ]]; then source .env.local; fi
export CXX="${GS_SEARCH_CXX:-/usr/bin/c++}"
if [[ -n "${GS_SEARCH_CC:-}" ]]; then export CC="$GS_SEARCH_CC"; else unset CC; fi
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.cache/torch_extensions}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MAX_JOBS="${MAX_JOBS:-4}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
exec "${GS_SEARCH_PYTHON:-python}" -m gs_search.cli "$@"
