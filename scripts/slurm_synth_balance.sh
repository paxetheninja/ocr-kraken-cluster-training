#!/bin/bash
# BALANCE variant of the synth ablation: oversample REAL in the real+synth mix so
# synth doesn't swamp the ~133 real inscriptions. The train manifest lists the real
# arrow REAL_REPEAT times + the synth_N arrow once → real counts R-fold.
# Same fixed real split (recog_train_real / recog_val) as slurm_synth_ablation.sh,
# eval letter-CER on the real val → directly comparable (R=1 == the plain ablation).
# Submit from inside the bundle dir:
#   BASE=/home/u2037/.local/share/htrmopo/d96caf7a-122e-5576-ab2b-a246c4e64221/catmus-print-fondue-large.mlmodel
#   sbatch --job-name=bal_n2500_r3 --export=ALL,N=2500,REAL_REPEAT=3,BASE_MODEL=$BASE ../slurm_synth_balance.sh
#   sbatch --job-name=bal_n2500_r5 --export=ALL,N=2500,REAL_REPEAT=5,BASE_MODEL=$BASE ../slurm_synth_balance.sh
#   sbatch --job-name=bal_n5000_r5 --export=ALL,N=5000,REAL_REPEAT=5,BASE_MODEL=$BASE ../slurm_synth_balance.sh
#
#SBATCH --job-name=edh-bal
#SBATCH --partition=compute
#SBATCH --gres=gpu:rtx6000bb:1
#SBATCH --account=ocr-inscriptions-0004
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -e
source "$HOME/kraken-venv/bin/activate"
mkdir -p logs cv models page_nospace
N="${N:?set N=2500 (number of synth)}"
REAL_REPEAT="${REAL_REPEAT:-3}"
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL=/path/to/catmus-print-fondue-large.mlmodel}"

# 1) space-free stripping (real train/val lists + first N synth); emit file lists
python3 - "$N" <<'PY'
import re, sys
from pathlib import Path
N = int(sys.argv[1])
PAGE = Path("page"); NS = Path("page_nospace"); NS.mkdir(exist_ok=True)
U = re.compile(r"(<Unicode>)([^<]*)(</Unicode>)")
def strip(b):
    s, d = PAGE / f"{b}.xml", NS / f"{b}.xml"
    if not s.exists():
        return None
    if not d.exists():
        d.write_text(U.sub(lambda m: m.group(1) + m.group(2).replace(' ', '') + m.group(3),
                           s.read_text(encoding='utf-8')), encoding='utf-8')
    return str(d)
def bases(lst):
    return [Path(x.strip()).stem for x in Path(lst).read_text().split() if x.strip()]
tr = [p for b in bases('lists/recog_train_real.txt') if (p := strip(b))]
va = [p for b in bases('lists/recog_val.txt')        if (p := strip(b))]
sy = [p for i in range(N)                            if (p := strip(f'synth_{i:05d}'))]
Path('cv/abl_real_train.list').write_text("\n".join(tr), encoding='utf-8')
Path('cv/abl_real_val.list').write_text("\n".join(va), encoding='utf-8')
Path(f'cv/abl_synth_{N}.list').write_text("\n".join(sy), encoding='utf-8')
print(f"real_train={len(tr)} real_val={len(va)} synth={len(sy)} (target {N})")
PY

# 2) compile arrows (reuse if already built by the ablation run)
[ -f cv/abl_tr_real.arrow ]    || ketos compile -f page -o cv/abl_tr_real.arrow    $(cat cv/abl_real_train.list)
[ -f cv/abl_va_real.arrow ]    || ketos compile -f page -o cv/abl_va_real.arrow    $(cat cv/abl_real_val.list)
[ -f cv/abl_synth_${N}.arrow ] || ketos compile -f page -o cv/abl_synth_${N}.arrow $(cat cv/abl_synth_${N}.list)

# 3) build the oversampled manifest: REAL_REPEAT copies of real + 1x synth
TAG=n${N}_r${REAL_REPEAT}
: > cv/bal_tr_${TAG}.manifest
for i in $(seq 1 "$REAL_REPEAT"); do echo cv/abl_tr_real.arrow >> cv/bal_tr_${TAG}.manifest; done
echo cv/abl_synth_${N}.arrow >> cv/bal_tr_${TAG}.manifest
echo cv/abl_va_real.arrow > cv/bal_va.manifest
echo "manifest (REAL_REPEAT=$REAL_REPEAT):"; cat cv/bal_tr_${TAG}.manifest

# 4) train
OUT=models/bal_${TAG}
rm -rf "$OUT"
ketos --workers 16 --threads 8 train -f binary \
    -t cv/bal_tr_${TAG}.manifest -e cv/bal_va.manifest \
    -i "$BASE_MODEL" -o "$OUT" -B 16 --resize new \
    -q early --min-epochs 15 --lag 10 -N 200

# 5) eval letter-CER on the real val
BEST=$(ls -t "$OUT"/best_*.safetensors | head -1)
ACC=$(ketos test -f binary -m "$BEST" cv/abl_va_real.arrow 2>&1 \
      | grep -i 'Character Accuracy' | grep -oE '[0-9.]+' | head -1)
echo "$ACC" > cv/result_bal_${TAG}.txt
echo "[balance N=$N R=$REAL_REPEAT] char_acc=$ACC  CER=$(python3 -c "print(f'{100-$ACC:.2f}')")%"
