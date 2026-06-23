#!/bin/bash
# Pretrain CATMuS-Print Large on the first N synthetic inscriptions, then export a
# CLEAN .mlmodel base for the 'pretrain' CV strategy. Parametrized by N + KEEP_DOTS
# + BUNDLE; run on compute (job cd's into the project bundle to trigger the autofs).
# Submitted by rerun_after_gtfix.sh as:
#   sbatch --chdir=$HOME --export=ALL,JOB=...,BUNDLE=...,KEEP_DOTS=...,BASE_MODEL=...,N=2500 \
#          /tank/projects/ocr-inscriptions-0004/edh/slurm_pretrain_synth.sh
# Output: models/pretrain_synth_N${N}/model.mlmodel  (base for STRATEGY=pretrain).
#
#SBATCH --job-name=edh-pre
#SBATCH --partition=compute
#SBATCH --gres=gpu:rtx6000bb:1
#SBATCH --account=ocr-inscriptions-0004
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -e
cd "${JOB:?set JOB=/mnt/nfs/projects/ocr-inscriptions-0004/edh}/${BUNDLE:-bundle}"
source "$HOME/kraken-venv/bin/activate"
mkdir -p logs cv models
N="${N:?set N (number of synth to pretrain on)}"
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL=/path/to/catmus-print-fondue-large.mlmodel}"
OUT="models/pretrain_synth_N${N}"

# 1) strip first N synth (KEEP_DOTS-aware) -> 90/10 pretrain train/val lists
python3 - "$N" <<'PY'
import os, re, sys
from pathlib import Path
N = int(sys.argv[1])
KEEP = os.environ.get("KEEP_DOTS", "0") == "1"
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
nval = max(1, len(sy) // 10)
Path(f'cv/pre_tr_N{N}.list').write_text("\n".join(sy[nval:]), encoding='utf-8')
Path(f'cv/pre_va_N{N}.list').write_text("\n".join(sy[:nval]), encoding='utf-8')
print(f"pretrain synth N={N}: train={len(sy)-nval} val={nval} [{NSDIR}]")
PY

# 2) compile arrows
ketos compile -f page -o cv/pre_tr_N${N}.arrow $(cat cv/pre_tr_N${N}.list)
ketos compile -f page -o cv/pre_va_N${N}.arrow $(cat cv/pre_va_N${N}.list)
echo cv/pre_tr_N${N}.arrow > cv/pre_tr_N${N}.manifest
echo cv/pre_va_N${N}.arrow > cv/pre_va_N${N}.manifest

# 3) pretrain CATMuS-Print Large on synth only
rm -rf "$OUT"
ketos --workers 16 --threads 8 train -f binary \
    -t cv/pre_tr_N${N}.manifest -e cv/pre_va_N${N}.manifest \
    -i "$BASE_MODEL" -o "$OUT" -B 16 --resize new \
    -q early --min-epochs 15 --lag 10 -N 100

# 4) export a coreml .mlmodel base from the BEST checkpoint. The raw
#    best_*.safetensors weights file is NOT loadable as a model (load_any only
#    reads coreml); `ketos convert --weights-format coreml` makes a usable -i base.
SCORE=$(ls "$OUT"/best_*.safetensors | sed -E 's/.*best_([0-9.]+)\.safetensors/\1/' | head -1)
CKPT=$(ls "$OUT"/checkpoint_*-${SCORE}.ckpt | head -1)
echo "best score=$SCORE -> $CKPT"
rm -f "$OUT/model.mlmodel"   # ketos convert refuses to overwrite an existing file
ketos convert --weights-format coreml -o "$OUT/model.mlmodel" "$CKPT"
ls -l "$OUT/model.mlmodel"
