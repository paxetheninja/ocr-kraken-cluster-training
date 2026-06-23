#!/bin/bash
# Simple real-only training as a smoke test — runs on CPU to bypass the broken
# libcuda on the Blackwell node. Proves the pipeline is intact (model loads,
# training runs) while the GPU is being fixed. Submit from the bundle dir:
#   sbatch ../slurm_smoke_real.sh
#
#SBATCH --job-name=smoke-real
#SBATCH --partition=compute
#SBATCH --account=ocr-inscriptions-0004
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -e
source "$HOME/kraken-venv/bin/activate"
mkdir -p logs cv models
BASE=/home/u2037/.local/share/htrmopo/d96caf7a-122e-5576-ab2b-a246c4e64221/catmus-print-fondue-large.mlmodel

# real-only fold0 (space-free), already prepared by cv_folds.py
ketos compile -f page -o cv/smoke_tr.arrow $(cat cv/fold0_train.txt)
ketos compile -f page -o cv/smoke_va.arrow $(cat cv/fold0_val.txt)
echo cv/smoke_tr.arrow > cv/smoke_tr.manifest
echo cv/smoke_va.arrow > cv/smoke_va.manifest

# -d cpu = force CPU (no GPU / libcuda needed)
ketos -d cpu --workers 16 train -f binary \
    -t cv/smoke_tr.manifest -e cv/smoke_va.manifest \
    -i "$BASE" -o models/smoke_real -B 1 --resize new \
    -N 20 -q early --min-epochs 10 --lag 8

BEST=$(ls -t models/smoke_real/best_*.safetensors | head -1)
ACC=$(ketos -d cpu test -f binary -m "$BEST" cv/smoke_va.arrow 2>&1 \
      | grep -i 'Character Accuracy' | grep -oE '[0-9.]+' | head -1)
echo "[smoke real] char_acc=$ACC  CER=$(python3 -c "print(f'{100-$ACC:.2f}')")%"
