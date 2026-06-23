#!/bin/bash
# Train a custom baseline+region SEGMENTATION model (blla-style) for the
# inscriptions, so eScriptorium auto-detects lines on new photos.
# Needs >=16 GB VRAM (blocked on the 8 GB local card); 96 GB is plenty.
# Submit from inside the bundle dir:  sbatch ../slurm_segmentation.sh
#
#SBATCH --job-name=edh-seg
#SBATCH --partition=compute           # dhinfra: rtx6000bb GPUs live on compute
#SBATCH --gres=gpu:rtx6000bb:1
#SBATCH --account=ocr-inscriptions-0004
#SBATCH --cpus-per-task=48            # node has 192 cores; segtrain is CPU-bound
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -e

source "$HOME/kraken-venv/bin/activate"
mkdir -p logs models

# ketos segtrain learns baselines + regions from our PAGE XML (Coords + Baseline).
# -i blla fine-tunes kraken's default baseline segmenter (recommended over scratch).
# If 'blla' is not resolvable by name, drop -i to train from scratch, or pass the
# path to kraken's bundled blla.mlmodel.
# SEG_LIST=lists/seg.txt for real-only; seg_all.txt = real + synthetic
SEG_LIST="${SEG_LIST:-lists/seg_all.txt}"
OUT="${OUT:-models/edh_seg}"                 # set OUT=models/edh_seg_real for real-only
# kraken's bundled blla segmenter: resolve its path (not resolvable by name here);
# fall back to training from scratch if not found.
BLLA=$(find "$HOME/kraken-venv" -iname 'blla*.mlmodel' 2>/dev/null | head -1)
LOAD=${BLLA:+-i "$BLLA"}
echo "blla base: ${BLLA:-<none, training from scratch>}"
ketos --workers 48 --threads 8 segtrain -f page $LOAD \
    -o "$OUT" \
    --resize union \
    -q early --min-epochs 20 --lag 15 -N 300 \
    $(cat "$SEG_LIST")

echo "Best segmentation model: $(ls -t models/edh_seg/*.mlmodel 2>/dev/null | head -1)"
echo "Use in eScriptorium: upload the .mlmodel as a Segmentation model."
