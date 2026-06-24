#!/bin/bash
# Isolate the middot tax: re-evaluate the MIT-middot (bundle_dots) trained CV models
# with the middot STRIPPED (letter-only) and compare to their middot-counted CER and
# to the ohne-middot models. If letter-only ~= ohne-middot -> learning the middot did
# NOT hurt the letters; the whole tax is interpunct placement.
# Run on the HEAD node (CPU eval, no GPU):
#   bash /tank/projects/ocr-inscriptions-0004/edh/eval_dots_letteronly.sh
WORK="$(cd "$(dirname "$0")" && pwd)"
D="$WORK/bundle_dots"
cd "$D" || { echo "no bundle_dots"; exit 1; }
source "$HOME/kraken-venv/bin/activate"
STRATS="${STRATS:-n5000_r5 real}"

echo "=== pre-flight: contents of one cv3 model dir (need a .ckpt OR a .safetensors to convert) ==="
ls -1 models/cvN_n5000_r5_0/ 2>/dev/null | sed 's/^/   /'
echo ""

evalone () {  # $1=label  $2=model-dir-prefix
  : > "cv/letteronly_$1.txt"
  for K in 0 1 2 3 4; do
    M="models/$2_$K"
    SCORE=$(ls "$M"/best_*.safetensors 2>/dev/null | sed -E 's/.*best_([0-9.]+)\.safetensors/\1/' | head -1)
    CKPT=$(ls "$M"/checkpoint_*-"${SCORE}".ckpt 2>/dev/null | head -1)
    SRC="${CKPT:-$(ls -t "$M"/best_*.safetensors 2>/dev/null | head -1)}"
    if [ -z "$SRC" ]; then echo "   $1 fold $K: no model file found"; continue; fi
    rm -f "$M/eval.mlmodel"
    if ! ketos convert --weights-format coreml -o "$M/eval.mlmodel" "$SRC" >/dev/null 2>&1; then
      echo "   $1 fold $K: ketos convert FAILED from $(basename "$SRC")"; continue
    fi
    CER=$(KEEP_DOTS=0 python "$WORK/eval_baseline.py" "$M/eval.mlmodel" "cv/fold${K}_val.txt" cpu 2>/dev/null \
          | grep -oE "letter-CER = [0-9.]+" | grep -oE "[0-9.]+")
    echo "   $1 fold $K  letter-only CER=$CER"
    [ -n "$CER" ] && echo "$CER" >> "cv/letteronly_$1.txt"
    rm -f "$M/eval.mlmodel"
  done
  python3 - "$1" <<'PY'
import os, statistics as st, sys
f=f"cv/letteronly_{sys.argv[1]}.txt"
v=[float(x) for x in open(f).read().split()] if os.path.exists(f) else []
if v: print(f">>> {sys.argv[1]}: mit-middot model, LETTER-ONLY CER = {st.mean(v):.2f} +/- {st.pstdev(v):.2f}  (n={len(v)})  {v}")
else: print(f">>> {sys.argv[1]}: no results")
PY
}

for S in $STRATS; do
  case "$S" in
    real|pretrain|mixed) evalone "$S" "cv_$S" ;;
    *)                   evalone "$S" "cvN_$S" ;;
  esac
done

echo ""
echo "=== comparison (letter-CER) ==="
printf "%-12s %8s %18s %18s\n" "strategy" "ohne·" "mit·(·-counted)" "mit·(letter-only)"
lo () { python3 -c "import statistics as st;v=[float(x) for x in open('cv/letteronly_$1.txt').read().split()];print(f'{st.mean(v):.2f}')" 2>/dev/null || echo '--'; }
printf "%-12s %8s %18s %18s\n" "n5000_r5" "15.32" "20.11" "$(lo n5000_r5)"
printf "%-12s %8s %18s %18s\n" "real"     "18.32" "22.94" "$(lo real)"
