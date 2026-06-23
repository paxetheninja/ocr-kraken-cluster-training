#!/bin/bash
# Re-measure recognition on the CORRECTED GT (letter-only: spaces AND · stripped).
# Data lives in the PROJECT dir. The project dataset is mounted at two paths:
#   head node:    /tank/projects/<proj>/...        (ZFS, login-side prep)
#   compute node: /mnt/nfs/projects/<proj>/...     (NFS rw, SLURM jobs run here)
# Layout next to this script (in <proj>/edh/):
#   bundle/{page,images,lists,cv,logs}  cv_folds.py  eval_baseline.py
#   slurm_cv.sh  slurm_cv_synthN.sh   (+ optional bundle/corrected_page.tar)
# Usage (run on the HEAD node):
#   bash /tank/projects/ocr-inscriptions-0004/edh/rerun_after_gtfix.sh
set -e
WORK="$(cd "$(dirname "$0")" && pwd)"                       # login-side path (head)
JOB="${WORK/#\/tank\/projects/\/mnt\/nfs\/projects}"        # compute-side path (NFS)
# Parameters (defaults = the full "ohne ·" grid; override for "mit ·" / subsets):
#   BUNDLE=bundle       data dir under <proj>/edh   (bundle | bundle_dots)
#   KEEP_DOTS=0         0 = strip spaces AND · ;  1 = keep · (strip spaces only)
#   PREFIX=cv2          SLURM job-name prefix (use a distinct one per condition)
#   STRATS="..."        CV grid: "real" or "N:R" tokens (N synth, real ×R)
#   PRETRAIN_NS="..."   synth-pretrain quantities ("" disables pretrain)
#   SKIP_STOCK=0        1 = skip the zero-shot stock baseline (already measured)
BUNDLE="${BUNDLE:-bundle}"
export KEEP_DOTS="${KEEP_DOTS:-0}"
PREFIX="${PREFIX:-cv2}"
STRATS="${STRATS:-real 500:1 2500:1 5000:1 5000:3 5000:5 5000:8}"
PRETRAIN_NS="${PRETRAIN_NS:-500 2500 5000}"
SKIP_STOCK="${SKIP_STOCK:-0}"
export WORK JOB BUNDLE
export BASE="$HOME/.local/share/htrmopo/d96caf7a-122e-5576-ab2b-a246c4e64221/catmus-print-fondue-large.mlmodel"
cd "$WORK/$BUNDLE"
source "$HOME/kraken-venv/bin/activate"
mkdir -p logs cv models
echo "WORK=$WORK  JOB=$JOB  BUNDLE=$BUNDLE  KEEP_DOTS=$KEEP_DOTS  PREFIX=$PREFIX"

echo "=== 1) backup + apply corrected real page XML (if tar present) ==="
mkdir -p page_pre_gtfix && cp -f page/HD*.xml page_pre_gtfix/ 2>/dev/null || true
[ -f corrected_page.tar ] && { tar -xf corrected_page.tar && echo "   applied corrected_page.tar"; }
echo "   real=$(ls page/HD*.xml | wc -l)  synth=$(ls page/synth_*.xml 2>/dev/null | wc -l)"

echo "=== 2) rebuild strip-dir + folds (HD-grouped; KEEP_DOTS=$KEEP_DOTS) ==="
python "$WORK/cv_folds.py"

if [ "$SKIP_STOCK" != 1 ]; then
echo "=== 3) stock baseline cross-fold (CPU; cd in-job triggers the NFS automount) ==="
srun -p compute -A ocr-inscriptions-0004 --chdir=/tmp --cpus-per-task=8 --mem=16G -t 00:40:00 bash -c '
  cd "$JOB/$BUNDLE" || { echo "FATAL: cannot cd $JOB/$BUNDLE on $(hostname)"; exit 1; }
  source "$HOME/kraken-venv/bin/activate"
  : > cv/stock_folds.txt
  for k in 0 1 2 3 4; do
    c=$(python ../eval_baseline.py "$BASE" cv/fold${k}_val.txt cpu | grep -oE "letter-CER = [0-9.]+" | grep -oE "[0-9.]+")
    echo "   fold $k CER=$c"; echo "$c" >> cv/stock_folds.txt
  done'
python - "$BUNDLE" "$KEEP_DOTS" <<'PY'
import statistics as st, sys
v=[float(x) for x in open("cv/stock_folds.txt").read().split()]
print(f"STOCK [{sys.argv[1]} KEEP_DOTS={sys.argv[2]}] cross-fold: "
      f"mean={st.mean(v):.2f}  std={st.pstdev(v):.2f}  folds={v}")
PY
fi

echo "=== 4) submit CV grid (cd in-job triggers automount; logs -> \$HOME/edh_logs) ==="
LOGDIR="$HOME/edh_logs"; mkdir -p "$LOGDIR"
LOG="--output=$LOGDIR/%x_%A_%a.out --error=$LOGDIR/%x_%A_%a.err"
EXP="ALL,JOB=$JOB,BUNDLE=$BUNDLE,KEEP_DOTS=$KEEP_DOTS,BASE_MODEL=$BASE"
for s in $STRATS; do
  if [ "$s" = real ]; then
    sbatch --chdir="$HOME" $LOG --job-name=${PREFIX}-real --export=$EXP,STRATEGY=real "$WORK/slurm_cv.sh"
  else
    N=${s%:*}; R=${s#*:}
    sbatch --chdir="$HOME" $LOG --job-name=${PREFIX}-n${N}r${R} \
           --export=$EXP,N=$N,REAL_REPEAT=$R "$WORK/slurm_cv_synthN.sh"
  fi
done

# pretrain grid: pretrain on N synth (-> .mlmodel) THEN CV-finetune on real (afterok)
for M in $PRETRAIN_NS; do
  pj=$(sbatch --parsable --chdir="$HOME" $LOG --job-name=${PREFIX}-pre${M} \
        --export=$EXP,N=$M "$WORK/slurm_pretrain_synth.sh")
  sbatch --chdir="$HOME" $LOG --job-name=${PREFIX}-pretrain${M} --dependency=afterok:$pj \
        --export=$EXP,STRATEGY=pretrain,PRETRAIN_BASE=models/pretrain_synth_N${M}/model.mlmodel \
        "$WORK/slurm_cv.sh"
done

squeue -u $USER -o "%.12i %.16j %.8T %.10M %R"
echo "Done. CV acc is in the best_<acc>.safetensors filenames (logs in \$HOME/edh_logs)."
