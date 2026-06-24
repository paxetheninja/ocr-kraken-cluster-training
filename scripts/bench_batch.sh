#!/bin/bash
# Batch-size sweep — find the throughput knob. bench_io.sh showed the job is
# OVERHEAD-bound (tiny 5.7M model, B=16), not IO-bound. A larger batch amortises
# the per-step host overhead -> should raise GPU util and cut wall-clock. This
# trains a fixed number of epochs on identical data at -B 16/32/64/128 and reports
# time + avg GPU-util per batch size.
#
# TIMING/UTIL ONLY -> models discarded. A larger batch CHANGES the CER for a real
# run (fewer gradient updates) and would need LR re-scaling + CV re-validation;
# here we only measure speed/util, not accuracy.
#
# Live phase: cat $HOME/edh_logs/benchbatch_status_<jobid>.txt
# Submit:
#   sbatch --chdir=$HOME --output=$HOME/edh_logs/%x_%j.out --error=$HOME/edh_logs/%x_%j.err \
#     --export=ALL,JOB=$JOB,BUNDLE=bundle,BASE_MODEL=$BASE,N=1000,EPOCHS=8,BATCHES="16 32 64 128" \
#     /tank/projects/ocr-inscriptions-0004/edh/bench_batch.sh
#SBATCH --job-name=bench-batch
#SBATCH --partition=compute
#SBATCH --gres=gpu:rtx6000bb:1
#SBATCH --account=ocr-inscriptions-0004
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -e
cd "${JOB:?set JOB=/mnt/nfs/projects/ocr-inscriptions-0004/edh}/${BUNDLE:-bundle}"
source "$HOME/kraken-venv/bin/activate"
BASE="${BASE_MODEL:?set BASE_MODEL=/path/to/catmus-print-fondue-large.mlmodel}"
N="${N:-1000}"; E="${EPOCHS:-8}"; K=0
BATCHES="${BATCHES:-16 32 64 128}"
WORKTMP=/tmp/benchb_$SLURM_JOB_ID
STATUS="$HOME/edh_logs/benchbatch_status_${SLURM_JOB_ID}.txt"
mkdir -p "$WORKTMP" "$HOME/edh_logs"; trap 'rm -rf "$WORKTMP"' EXIT
echo "host=$(hostname)  N=$N  epochs=$E  batches=[$BATCHES]"

say () { echo "### $* ###"; echo "$(date +%H:%M:%S)  $*" > "$STATUS"; }

# robust per-sample GPU util of OUR GPU (cgroup-isolated -> only one GPU visible)
sample_util () {
  local out n
  out=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  n=$(printf '%s\n' "$out" | grep -c '[0-9]')
  if [ "${n:-0}" -le 1 ]; then printf '%s\n' "$out" | awk -F', *' 'NF>=2{print $2}'
  else printf '%s\n' "$out" | awk -F', *' -v g="${CUDA_VISIBLE_DEVICES%%,*}" '$1==g{print $2}'; fi
}

# first-N synth list (idempotent; page_nospace already populated by cv_folds)
python3 - "$N" <<'PY'
import re, sys
from pathlib import Path
N = int(sys.argv[1]); PAGE = Path("page"); NS = Path("page_nospace"); NS.mkdir(exist_ok=True)
U = re.compile(r"(<Unicode>)([^<]*)(</Unicode>)")
def strip(b):
    s, d = PAGE / f"{b}.xml", NS / f"{b}.xml"
    if not s.exists(): return None
    if not d.exists():
        d.write_text(U.sub(lambda m: m.group(1) + m.group(2).replace(' ', '').replace('·', '') + m.group(3),
                           s.read_text(encoding='utf-8')), encoding='utf-8')
    return str(d)
Path(f'cv/bench_synth_{N}.txt').write_text(
    "\n".join(p for i in range(N) if (p := strip(f'synth_{i:05d}'))), encoding='utf-8')
PY

TR="$(cat cv/fold${K}_train.txt) $(cat cv/bench_synth_${N}.txt)"
VA="$(cat cv/fold${K}_val.txt)"

say "compile (one-time)"
ketos compile -f page -o cv/benchb_tr.arrow $TR 1>&2
ketos compile -f page -o cv/benchb_va.arrow $VA 1>&2
echo cv/benchb_tr.arrow > cv/benchb_tr.manifest
echo cv/benchb_va.arrow > cv/benchb_va.manifest

RES=()
for B in $BATCHES; do
  say "training -B $B (RUNNING)"
  u="$WORKTMP/util_$B.csv"
  ( while :; do sample_util; sleep 2; done ) >"$u" 2>/dev/null & samp=$!
  t=$SECONDS
  if ketos --workers 4 --threads 8 train -f binary -t cv/benchb_tr.manifest -e cv/benchb_va.manifest \
       -i "$BASE" -o "$WORKTMP/b$B" -B "$B" --resize new -q fixed -N "$E" 1>&2; then
    d=$((SECONDS-t))
  else
    d=-1   # failed (e.g. OOM)
  fi
  kill "$samp" 2>/dev/null || true
  a=$(awk '{s+=$1;n++} END{if(n)printf "%.0f",s/n; else print "NA"}' "$u")
  RES+=("$B|$d|$a")
  rm -rf "$WORKTMP/b$B"
done
say "DONE"

echo ""
echo "==================== BATCH SWEEP (N=$N, ${E} epochs) ===================="
printf "%-7s %9s %7s %9s\n" "batch" "time" "util" "vs B16"
base=""
for r in "${RES[@]}"; do
  B=${r%%|*}; rest=${r#*|}; d=${rest%%|*}; a=${rest#*|}
  if [ "$d" = "-1" ]; then printf "%-7s %9s %7s%% %9s\n" "$B" "FAILED" "$a" "-"; continue; fi
  [ -z "$base" ] && base=$d
  sp=$(awk "BEGIN{if($d>0)printf \"%.2fx\",$base/$d; else print \"--\"}")
  printf "%-7s %8ss %6s%% %9s\n" "$B" "$d" "$a" "$sp"
done
echo "(timing/util only; larger batch changes CER -> would need LR scaling + CV re-check for a real run)"
rm -f "$STATUS"
