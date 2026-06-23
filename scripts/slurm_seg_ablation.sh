#!/bin/bash
# Segmentation (blla baseline+region) trained on real seg pages + the first N synth
# page-XML. Mirrors slurm_segmentation.sh but with a synth-count knob and 70 CPUs.
# Segmentation is geometry-only → uses the WITH-spaces page/ XML (baselines + regions).
# Submit from inside the bundle dir:
#   sbatch --job-name=seg2500 --export=ALL,N=2500 ../slurm_seg_ablation.sh
#   sbatch --job-name=seg5000 --export=ALL,N=5000 ../slurm_seg_ablation.sh
#
#SBATCH --job-name=edh-segN
#SBATCH --partition=compute
#SBATCH --gres=gpu:rtx6000bb:1
#SBATCH --account=ocr-inscriptions-0004
#SBATCH --cpus-per-task=70
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -e
source "$HOME/kraken-venv/bin/activate"
mkdir -p logs models cv
N="${N:?set N=2500 (number of synth)}"
REAL_VAL="${REAL_VAL:-50}"   # size of the fixed real validation set

# Train = real train pages + first N synth page-XML.
# Val   = REAL_VAL real pages ONLY (fixed): the 24 recog_val + extra from the
# train pool, deterministic, no leak -> honest, stable, comparable F1 across N.
python3 - "$N" "$REAL_VAL" <<'PY'
import sys
from pathlib import Path
N, VALN = int(sys.argv[1]), int(sys.argv[2])
def stems(lst):
    return [Path(x.strip()).stem for x in Path(lst).read_text().split() if x.strip()]
def page(s): return f"page/{s}.xml"
val_core = stems('lists/recog_val.txt')          # always held out
pool     = sorted(stems('lists/recog_train_real.txt'))
extra    = set(pool[:max(0, VALN - len(val_core))])
val_stems   = val_core + sorted(extra)
train_stems = [s for s in pool if s not in extra]
val     = [page(s) for s in val_stems   if Path(page(s)).exists()]
real_tr = [page(s) for s in train_stems if Path(page(s)).exists()]
synth   = [f'page/synth_{i:05d}.xml' for i in range(N) if Path(f'page/synth_{i:05d}.xml').exists()]
Path(f'cv/seg_train_n{N}.txt').write_text("\n".join(real_tr + synth), encoding='utf-8')
Path('cv/seg_val_real.txt').write_text("\n".join(val), encoding='utf-8')
print(f"real_train={len(real_tr)} synth={len(synth)} train_total={len(real_tr)+len(synth)} | real_val={len(val)}")
PY

# fine-tune kraken's bundled blla segmenter if resolvable, else train from scratch
BLLA=$(find "$HOME/kraken-venv" -iname 'blla*.mlmodel' 2>/dev/null | head -1)
LOAD=${BLLA:+-i "$BLLA"}
echo "blla base: ${BLLA:-<none, from scratch>}"

OUT=models/edh_seg_n${N}
rm -rf "$OUT"
# -e overrides the auto-partition -> validate on the fixed real val only
ketos --workers 70 --threads 8 segtrain -f page $LOAD \
    -o "$OUT" --resize union \
    -e cv/seg_val_real.txt \
    -q early --min-epochs 20 --lag 15 -N 300 \
    $(cat cv/seg_train_n${N}.txt)

echo "best segmentation model: $(ls -t "$OUT"/*.mlmodel "$OUT"/*.safetensors 2>/dev/null | head -1)"
echo "Baseline F1 on the REAL val is in the val_metric / 'Accuracy' lines above."
