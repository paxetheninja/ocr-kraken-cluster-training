#!/bin/bash
# One-time environment setup on the cluster (run on a login node, NOT via sbatch).
# Creates a kraken venv and shows how to fetch the CATMuS-Print base models.
set -e

# --- ADJUST to your cluster's module system (or skip if python/CUDA already on PATH) ---
# module load python/3.11
# module load cuda/12.1

VENV="$HOME/kraken-venv"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel
pip install "kraken>=5.2"          # pulls a CUDA-enabled torch

echo
echo "kraken installed. GPU check:"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo
echo "=== Fetch base model: CATMuS-Print LARGE ==="
kraken get 10.5281/zenodo.10592716     # -> catmus-print-fondue-large.mlmodel
LARGE=$(find "$HOME/.local/share/htrmopo" -iname '*fondue-large*.mlmodel' | head -1)
echo "CATMuS-Print Large at: $LARGE"
echo "Export it before submitting recognition:  export BASE_MODEL=$LARGE"
# Segmentation base: kraken ships the default 'blla' baseline model; ketos segtrain
# fine-tunes it (referenced as 'blla' in slurm_segmentation.sh).
