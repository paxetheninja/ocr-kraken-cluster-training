#!/bin/bash
# 5-fold group-CV of CATMuS-Print Large with the real fold + first N synth,
# optional real oversampling (REAL_REPEAT). Eval = real fold val (space-free).
# acc is read from the best_<acc>.safetensors filename (reliable; ketos-test grep
# was unreliable). Result per fold -> cv/result_cvN_<tag>.txt
#
#   BASE=/home/u2037/.local/share/htrmopo/d96caf7a-122e-5576-ab2b-a246c4e64221/catmus-print-fondue-large.mlmodel
#   sbatch --job-name=cvN2500   --export=ALL,N=2500,BASE_MODEL=$BASE ../slurm_cv_synthN.sh
#   sbatch --job-name=cvN5000   --export=ALL,N=5000,BASE_MODEL=$BASE ../slurm_cv_synthN.sh
#   # balanced (later):
#   sbatch --job-name=cvN5000r5 --export=ALL,N=5000,REAL_REPEAT=5,BASE_MODEL=$BASE ../slurm_cv_synthN.sh
#
#SBATCH --job-name=edh-cvN
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
mkdir -p logs cv models page_nospace
K=$SLURM_ARRAY_TASK_ID
N="${N:?set N=2500 (number of synth)}"
REAL_REPEAT="${REAL_REPEAT:-1}"
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL=/path/to/catmus-print-fondue-large.mlmodel}"
TAG=n${N}_r${REAL_REPEAT}_$K

# 1) strip first N synth to nospace + build list
python3 - "$N" <<'PY'
import os, re, sys
from pathlib import Path
N = int(sys.argv[1])
KEEP = os.environ.get("KEEP_DOTS", "0") == "1"          # 1 -> keep ·, write page_dots/
NSDIR = "page_dots" if KEEP else "page_nospace"
PAGE = Path("page"); NS = Path(NSDIR); NS.mkdir(exist_ok=True)
U = re.compile(r"(<Unicode>)([^<]*)(</Unicode>)")
def clean(s):
    s = s.replace(' ', '')
    return s if KEEP else s.replace('·', '')
def strip(b):
    s, d = PAGE / f"{b}.xml", NS / f"{b}.xml"
    if not s.exists(): return None
    if not d.exists():
        d.write_text(U.sub(lambda m: m.group(1) + clean(m.group(2)) + m.group(3),
                           s.read_text(encoding='utf-8')), encoding='utf-8')
    return str(d)
sy = [p for i in range(N) if (p := strip(f'synth_{i:05d}'))]
Path(f'cv/synth_n{N}.txt').write_text("\n".join(sy), encoding='utf-8')
print(f"synth available: {len(sy)} (target {N}) [{NSDIR}]")
PY

# 2) compile arrows into RAM-backed scratch (tmpfs): training reads from RAM, not
#    per-batch over NFS (the IO bottleneck). The per-task RAMDIR is private, so it
#    also removes any arrow-name collision between concurrent grid jobs. Neutral.
RAMDIR=/dev/shm/$USER/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$RAMDIR" 2>/dev/null || RAMDIR=$(mktemp -d)
trap 'rm -rf "$RAMDIR"' EXIT
ketos compile -f page -o "$RAMDIR/tr_real.arrow" $(cat cv/fold${K}_train.txt)
ketos compile -f page -o "$RAMDIR/va.arrow"      $(cat cv/fold${K}_val.txt)
ketos compile -f page -o "$RAMDIR/synth.arrow"   $(cat cv/synth_n${N}.txt)

# 3) manifest: REAL_REPEAT copies of the real fold + 1x synth
: > "$RAMDIR/tr.manifest"
for i in $(seq 1 "$REAL_REPEAT"); do echo "$RAMDIR/tr_real.arrow" >> "$RAMDIR/tr.manifest"; done
echo "$RAMDIR/synth.arrow" >> "$RAMDIR/tr.manifest"
echo "$RAMDIR/va.arrow" > "$RAMDIR/va.manifest"

# 4) train
OUT=models/cvN_$TAG
rm -rf "$OUT"
ketos --workers 8 --threads 8 train -f binary \
    -t "$RAMDIR/tr.manifest" -e "$RAMDIR/va.manifest" \
    -i "$BASE_MODEL" -o "$OUT" -B 16 --resize new \
    -q early --min-epochs 20 --lag 15 -N 300

# 5) acc from the best_<acc>.safetensors filename (fraction, e.g. 0.8420)
BEST=$(ls -t "$OUT"/best_*.safetensors | head -1)
ACC=$(echo "$BEST" | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "$ACC" > cv/result_cvN_$TAG.txt
echo "[cv N=$N R=$REAL_REPEAT fold $K] val_acc=$ACC  CER=$(awk "BEGIN{printf \"%.2f\",(1-$ACC)*100}")%"
