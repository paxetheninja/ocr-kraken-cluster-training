#!/bin/bash
# Fine-tune CATMuS-Print LARGE on the EDH inscriptions.
# Our best local config (8 GB card): Large + REAL only + scriptio continua
# -> 14.53% letter-CER (synth helped the Small base but HURTS the Large base).
# The 96 GB GPU lets you push it: bigger batch, more epochs, optionally + synth.
#
# Submit from inside the bundle dir:  sbatch ../slurm_recognition.sh
#
#SBATCH --job-name=edh-recog
#SBATCH --partition=compute           # dhinfra: rtx6000bb GPUs live on compute
#SBATCH --gres=gpu:rtx6000bb:1
#SBATCH --account=ocr-inscriptions-0004
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -e

source "$HOME/kraken-venv/bin/activate"
mkdir -p logs models data

# CATMuS-Print Large (download once: kraken get 10.5281/zenodo.10592716)
BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL=/path/to/catmus-print-fondue-large.mlmodel}"

# choose training set: real-only (best) or real+synth
TRAIN_LIST="${TRAIN_LIST:-lists/recog_train_real.txt}"   # or lists/recog_train.txt (+synth)

# optional: strip spaces for scriptio-continua / letter-CER (our best setup)
if [ "${STRIP_SPACES:-1}" = "1" ]; then
  python - "$TRAIN_LIST" lists/recog_val.txt <<'PY'
import re, sys
for lf in sys.argv[1:]:
    for rel in open(lf).read().split():
        p=rel.strip()
        if not p: continue
        t=re.sub(r'(<Unicode>)([^<]*)(</Unicode>)', lambda m:m.group(1)+m.group(2).replace(' ','')+m.group(3), open(p,encoding='utf-8').read())
        open(p,'w',encoding='utf-8').write(t)
print("stripped spaces from GT")
PY
fi

ketos compile -f page -o data/train.arrow $(cat "$TRAIN_LIST")
ketos compile -f page -o data/val.arrow   $(cat lists/recog_val.txt)

# this kraken wants a MANIFEST (text file listing the arrow files) for -t/-e
echo data/train.arrow > data/train.manifest
echo data/val.arrow   > data/val.manifest

ketos --workers 8 --threads 8 train -f binary \
    -t data/train.manifest -e data/val.manifest \
    -i "$BASE_MODEL" -o models/edh_recog \
    -B 16 --resize new \
    -q early --min-epochs 20 --lag 15 -N 300

echo "Best: $(ls -t models/edh_recog/*.mlmodel 2>/dev/null | head -1)"
echo "Eval: ketos test -f binary -m <best.mlmodel> data/val.arrow"
