#!/bin/bash
# A/B IO benchmark — quantify the A+B optimisation (RAM arrows + workers).
# Times AND measures GPU-util for three configs on the same data + same fixed
# epoch count:
#   compile  (NFS images -> arrow; one-time small-file IO)
#   A: train from NFS arrow,  --workers 4   (OLD baseline)
#   B: train from RAM arrow,  --workers 4   (A->B isolates the RAM effect)
#   C: train from RAM arrow,  --workers 8   (B->C isolates the workers effect)
# Live phase shown in $HOME/edh_logs/bench_status_<jobid>.txt  (cat it anytime).
# Submit:
#   sbatch --chdir=$HOME --output=$HOME/edh_logs/%x_%j.out --error=$HOME/edh_logs/%x_%j.err \
#     --export=ALL,JOB=$JOB,BUNDLE=bundle,BASE_MODEL=$BASE,N=5000,EPOCHS=4 \
#     /tank/projects/ocr-inscriptions-0004/edh/bench_io.sh
#SBATCH --job-name=bench-io
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
N="${N:-2500}"; E="${EPOCHS:-3}"; K=0
WORKTMP=/tmp/bench_$SLURM_JOB_ID; RAM=/dev/shm/$USER/bench_$SLURM_JOB_ID
STATUS="$HOME/edh_logs/bench_status_${SLURM_JOB_ID}.txt"
mkdir -p "$WORKTMP" "$RAM" "$HOME/edh_logs"; trap 'rm -rf "$WORKTMP" "$RAM"' EXIT
echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  N=$N  epochs=$E  bundle=$BUNDLE"
df -h /dev/shm | tail -1

say () { echo "### $* ###"; echo "$(date +%H:%M:%S)  $*" > "$STATUS"; }

# Robust per-sample GPU util of OUR GPU: a cgroup-isolated job sees only one GPU
# (use it); on a shared view, filter by CUDA_VISIBLE_DEVICES. Prints one %/call.
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

say "compile (NFS images -> arrow)"
t=$SECONDS
ketos compile -f page -o cv/bench_tr.arrow $TR 1>&2
ketos compile -f page -o cv/bench_va.arrow $VA 1>&2
C=$((SECONDS-t))

echo cv/bench_tr.arrow > cv/nfs_tr.manifest; echo cv/bench_va.arrow > cv/nfs_va.manifest
cp cv/bench_tr.arrow cv/bench_va.arrow "$RAM"/
echo "$RAM/bench_tr.arrow" > "$RAM/tr.manifest"; echo "$RAM/bench_va.arrow" > "$RAM/va.manifest"

# FIXED-epoch training: -q fixed -N E runs EXACTLY E epochs (deterministic, fair
# A/B/C comparison). NOTE: -q early IGNORES -N as a hard cap and stops only after
# --lag non-improving epochs -> with a large lag it trains ~forever (that hit the
# time limit). Use fixed mode for a benchmark. Background util sampler on OUR GPU;
# returns "seconds|avg_util%".
train () {  # $1=label $2=tr.manifest $3=va.manifest $4=workers
  local u="$WORKTMP/util_$1.csv"
  ( while :; do sample_util; sleep 2; done ) >"$u" 2>/dev/null &
  local samp=$!
  local t=$SECONDS
  ketos --workers "$4" --threads 8 train -f binary -t "$2" -e "$3" \
      -i "$BASE" -o "$WORKTMP/$1" -B 16 --resize new \
      -q fixed -N "$E" 1>&2
  local d=$((SECONDS-t))
  kill "$samp" 2>/dev/null || true
  local a=$(awk '{s+=$1;n++} END{if(n)printf "%.0f",s/n; else print "NA"}' "$u")
  echo "${d}|${a}"
}

say "A: NFS arrow, workers 4 (RUNNING)"; rA=$(train A cv/nfs_tr.manifest cv/nfs_va.manifest 4)
say "B: RAM arrow, workers 4 (RUNNING)"; rB=$(train B "$RAM/tr.manifest" "$RAM/va.manifest" 4)
say "C: RAM arrow, workers 8 (RUNNING)"; rC=$(train C "$RAM/tr.manifest" "$RAM/va.manifest" 8)
say "DONE"
A=${rA%|*}; uA=${rA#*|}; B=${rB%|*}; uB=${rB#*|}; Cc=${rC%|*}; uC=${rC#*|}

echo ""
echo "==================== BENCH RESULT (N=$N, ${E} epochs, GPU#$GIDX) ===================="
printf "compile (NFS images -> arrow):  %4ss\n" "$C"
printf "A  NFS arrow,  workers 4:        %4ss   util ~%s%%   (old baseline)\n" "$A"  "$uA"
printf "B  RAM arrow,  workers 4:        %4ss   util ~%s%%   (A->B = RAM effect)\n" "$B"  "$uB"
printf "C  RAM arrow,  workers 8:        %4ss   util ~%s%%   (B->C = workers effect)\n" "$Cc" "$uC"
awk "BEGIN{printf \"A->B  RAM:     %.2fx (%+.0f%%)\n\", $A/$B, (1-$B/$A)*100}"
awk "BEGIN{printf \"B->C  workers: %.2fx (%+.0f%%)\n\", $B/$Cc, (1-$Cc/$B)*100}"
awk "BEGIN{printf \"A->C  total:   %.2fx (%+.0f%%)\n\", $A/$Cc, (1-$Cc/$A)*100}"
echo "(timing only — trained models discarded)"
rm -f "$STATUS"
