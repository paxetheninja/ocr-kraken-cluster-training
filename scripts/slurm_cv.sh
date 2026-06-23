#!/bin/bash
# 5-fold group-CV of CATMuS-Print Large under one of three synth strategies.
# Pick the strategy at submit time via STRATEGY (real|mixed|pretrain):
#   sbatch --job-name=cv-real     --export=ALL,STRATEGY=real     ../slurm_cv.sh
#   sbatch --job-name=cv-mixed    --export=ALL,STRATEGY=mixed    ../slurm_cv.sh
#   sbatch --job-name=cv-pretrain --export=ALL,STRATEGY=pretrain ../slurm_cv.sh   # after slurm_pretrain_synth.sh
# Run python ../cv_folds.py first. Submit from the bundle dir.
#
#SBATCH --job-name=edh-cv
#SBATCH --array=0-4
#SBATCH --partition=compute
#SBATCH --gres=gpu:rtx6000bb:1
#SBATCH --account=ocr-inscriptions-0004
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -e
# Data lives in the PROJECT dir; compute nodes reach it via an NFS AUTOMOUNT at
# /mnt/nfs/projects/... . A plain `cd` (not srun/sbatch --chdir) triggers the
# automount reliably; --chdir fires too early and lands in / .
cd "${JOB:?set JOB=/mnt/nfs/projects/ocr-inscriptions-0004/edh}/${BUNDLE:-bundle}"
source "$HOME/kraken-venv/bin/activate"
mkdir -p logs cv models
K=$SLURM_ARRAY_TASK_ID
STRATEGY="${STRATEGY:-real}"
BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL=/path/to/catmus-print-fondue-large.mlmodel}"

case "$STRATEGY" in
  real)     TRAIN_PAGES="$(cat cv/fold${K}_train.txt)";                          BASE="$BASE_MODEL" ;;
  mixed)    TRAIN_PAGES="$(cat cv/fold${K}_train.txt) $(cat cv/synth_all.txt)";  BASE="$BASE_MODEL" ;;
  pretrain) TRAIN_PAGES="$(cat cv/fold${K}_train.txt)";                          BASE="${PRETRAIN_BASE:-models/pretrain_synth/model.mlmodel}" ;;
  *) echo "unknown STRATEGY=$STRATEGY"; exit 1 ;;
esac
echo "strategy=$STRATEGY fold=$K base=$BASE"

TAG=${STRATEGY}_$K
# Compile into RAM-backed scratch (tmpfs): training then reads the dataset from
# RAM instead of per-batch over NFS (the IO bottleneck). Behaviour-neutral.
RAMDIR=/dev/shm/$USER/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$RAMDIR" 2>/dev/null || RAMDIR=$(mktemp -d)
trap 'rm -rf "$RAMDIR"' EXIT
ketos compile -f page -o "$RAMDIR/tr.arrow" $TRAIN_PAGES
ketos compile -f page -o "$RAMDIR/va.arrow" $(cat cv/fold${K}_val.txt)
echo "$RAMDIR/tr.arrow" > "$RAMDIR/tr.manifest"
echo "$RAMDIR/va.arrow" > "$RAMDIR/va.manifest"

ketos --workers 8 --threads 8 train -f binary \
    -t "$RAMDIR/tr.manifest" -e "$RAMDIR/va.manifest" \
    -i "$BASE" -o models/cv_$TAG -B 16 --resize new \
    -q early --min-epochs 20 --lag 15 -N 300

BEST=$(ls -t models/cv_$TAG/best_*.safetensors | head -1)
ACC=$(echo "$BEST" | grep -oE '[0-9]+\.[0-9]+' | head -1)   # val acc from best_<acc>.safetensors
echo "$ACC" > cv/result_$TAG.txt
echo "[$STRATEGY fold $K] best=$BEST acc=$ACC CER=$(awk "BEGIN{printf \"%.2f\",(1-$ACC)*100}")%"
