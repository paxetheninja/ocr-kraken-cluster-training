#!/bin/bash
# Synth-QUANTITY ablation on the cluster: fine-tune CATMuS-Print Large on the
# fixed real train split + the first N synthetic inscriptions (space-free),
# eval letter-CER on the fixed real val (recog_val). Mirrors the local single
# split, so comparable to: real-only 14.53%, real+500synth 15.47%.
# Submit from inside the bundle dir AFTER uploading the synth (see header note):
#   BASE=/home/u2037/.local/share/htrmopo/d96caf7a-122e-5576-ab2b-a246c4e64221/catmus-print-fondue-large.mlmodel
#   sbatch --job-name=synth2500 --export=ALL,N=2500,BASE_MODEL=$BASE ../slurm_synth_ablation.sh
#   sbatch --job-name=synth5000 --export=ALL,N=5000,BASE_MODEL=$BASE ../slurm_synth_ablation.sh
#
#SBATCH --job-name=edh-synthN
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
        d.write_text(U.sub(lambda m: m.group(1) + m.group(2).replace(' ', '').replace('·', '') + m.group(3),
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

# 2) compile arrows
ketos compile -f page -o cv/abl_tr_real.arrow    $(cat cv/abl_real_train.list)
ketos compile -f page -o cv/abl_va_real.arrow    $(cat cv/abl_real_val.list)
ketos compile -f page -o cv/abl_synth_${N}.arrow $(cat cv/abl_synth_${N}.list)
printf 'cv/abl_tr_real.arrow\ncv/abl_synth_%s.arrow\n' "$N" > cv/abl_tr_${N}.manifest
echo cv/abl_va_real.arrow > cv/abl_va.manifest

# 3) train CATMuS-Print Large -> real + N synth
OUT=models/large_synth${N}
rm -rf "$OUT"
ketos --workers 16 --threads 8 train -f binary \
    -t cv/abl_tr_${N}.manifest -e cv/abl_va.manifest \
    -i "$BASE_MODEL" -o "$OUT" -B 16 --resize new \
    -q early --min-epochs 15 --lag 10 -N 200

# 4) eval letter-CER on the real val
BEST=$(ls -t "$OUT"/best_*.safetensors | head -1)
ACC=$(ketos test -f binary -m "$BEST" cv/abl_va_real.arrow 2>&1 \
      | grep -i 'Character Accuracy' | grep -oE '[0-9.]+' | head -1)
echo "$ACC" > cv/result_synth${N}.txt
echo "[synth N=$N] char_acc=$ACC  CER=$(python3 -c "print(f'{100-$ACC:.2f}')")%"
