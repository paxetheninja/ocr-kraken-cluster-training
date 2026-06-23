# OCR for Latin stone inscriptions — Kraken/ketos cluster training

SLURM training pipeline for a specialised HTR model on Latin epigraphic (stone
inscription) photos. Fine-tunes **CATMuS-Print Large** (Kraken) on a mix of real
EDH inscriptions and procedurally generated synthetic stone images, evaluated by
5-fold cross-validation. Built for the DHinfra cluster (Blackwell `rtx6000bb`).

This repo is the **training pipeline only** — a self-contained, reviewable slice
of a larger research project. The data (real GT, synthetic set, models) lives on
the cluster / in the research repo and is intentionally not committed here.

> **Why this repo exists right now:** to review and optimise the training
> throughput together. See [Performance notes](#performance-notes--known-bottlenecks)
> — that section is the starting point for the optimisation discussion.

---

## Layout

```
scripts/
  setup_env.sh             one-time venv + base-model fetch (login node)
  slurm_recognition.sh     canonical single fine-tune (real + synth) — simplest example
  slurm_segmentation.sh    blla baseline/region segmentation fine-tune
  slurm_cv.sh              5-fold group-CV, strategy-aware (real | mixed | pretrain)
  slurm_cv_synthN.sh       5-fold CV with N synth + real oversampling (REAL_REPEAT)
  slurm_pretrain_synth.sh  pretrain on synth -> export .mlmodel base for finetune
  rerun_after_gtfix.sh     orchestration: prep -> stock baseline -> submit the grid
  slurm_synth_ablation.sh  single-split synth-quantity ablation (N synth)
  slurm_synth_balance.sh   single-split synth + real-oversampling ablation
  slurm_seg_ablation.sh    segmentation ablation (fixed real val, varying synth)
  slurm_smoke_real.sh      quick smoke test on real data only
data_prep/
  cv_folds.py              build group-CV folds (group = inscription / HD number)
  eval_baseline.py         zero-shot letter-CER of any model on a PAGE-XML list
  bundle_for_cluster.py    assemble the portable bundle/ (paths from the research repo)
  make_tarball.sh          pack bundle + scripts into one archive to ship
synthetic/
  generate_synthetic.py    render synthetic stone images + PAGE XML from EDH texts
```

## Environment

```bash
bash scripts/setup_env.sh        # creates ~/kraken-venv, installs kraken, fetches CATMuS-Print Large
export BASE_MODEL=$HOME/.local/share/htrmopo/.../catmus-print-fondue-large.mlmodel
```
Stack: `kraken` (Lightning + PyTorch backend, CUDA build), CATMuS-Print **Large**
base (`10.5281/zenodo.10592716`, ~5.7 M params, 22 MB `.mlmodel`).

## Data layout (the "bundle")

All scripts run from a flat, relative `bundle/` directory:
```
bundle/
  page/<base>.xml        PAGE XML (imageFilename -> ../images/<base>.jpg)
  images/<base>.jpg      real photos + synthetic renders
  lists/recog_*.txt      train/val file lists
  cv/                    compiled arrows + fold lists + result_*.txt (generated)
  models/                training outputs (generated)
```
On the cluster the bundle lives in the **project dir** (100 G), which is mounted
at two paths: `/tank/projects/<proj>/…` on the head node (ZFS) and
`/mnt/nfs/projects/<proj>/…` on compute nodes (NFS-RDMA **autofs**). Jobs `cd`
into the bundle as their first action so the autofs mount triggers reliably —
`sbatch --chdir` onto the project path fires too early and lands in `/`.

## Training pipeline

1. **Compile** PAGE XML → Kraken binary dataset (arrow):
   `ketos compile -f page -o cv/x.arrow $(cat list.txt)`
2. **Train** (fine-tune the base):
   ```
   ketos --workers 4 --threads 8 train -f binary \
       -t train.manifest -e val.manifest \
       -i "$BASE_MODEL" -o models/out -B 16 --resize new \
       -q early --min-epochs 20 --lag 15 -N 300
   ```
3. **Score**: validation accuracy is read from the `best_<acc>.safetensors`
   filename; letter-CER = `(1 - acc) * 100`. (Text is scriptio continua: spaces
   stripped; the middot `·` optionally kept — see `KEEP_DOTS` in the scripts.)

Strategies are selected per submission via env vars (`STRATEGY`, `N`,
`REAL_REPEAT`, `KEEP_DOTS`, `BUNDLE`, `BASE_MODEL`). `rerun_after_gtfix.sh`
submits the whole comparison grid in one go.

```bash
# one fold-array job, real + 5000 synth, real oversampled x5:
sbatch --export=ALL,JOB=$JOB,BUNDLE=bundle,N=5000,REAL_REPEAT=5,BASE_MODEL=$BASE_MODEL scripts/slurm_cv_synthN.sh
```

## Performance notes / known bottlenecks

Current per-job config and what we've observed (this is the optimisation target):

| Aspect | Current | Observation |
|---|---|---|
| GPU | 1× `rtx6000bb` (Blackwell, ~97 GB) per array task | **~60 % util/card**, VRAM almost unused |
| Model | CATMuS-Print Large, ~5.7 M params | tiny relative to the card → compute underfed |
| Batch | `-B 16` | small; VRAM headroom is large |
| Dataloader | `--workers 4 --threads 8` | suspected **IO-bound** |
| Dataset | Kraken `binary` arrows on **NFS** (`/mnt/nfs/projects`, RDMA) | NFS read latency a likely factor |
| Precision | default fp32 | Kraken logs: *"set `torch.set_float32_matmul_precision('medium'|'high')`"* → Tensor Cores not fully used |
| Parallelism | one fold per GPU (array tasks), no intra-train DDP | — |

Candidate levers to discuss (not yet applied):
- **Larger batch** (VRAM is free) + **more dataloader workers** to feed the GPU.
- **Stage arrows to node-local scratch** before training to cut NFS latency.
- **`set_float32_matmul_precision('high')`** / mixed precision for Tensor Cores.
- Whether **2-GPU DDP** is even worth it for a 5.7 M-param model, vs. just
  packing more fold/array tasks per node.

## Result context (so the numbers mean something)

5-fold group-CV (group = inscription, real held out), corrected GT, letter-CER:

| | CER |
|---|---|
| zero-shot CATMuS-Print Large | 56.0 % |
| fine-tuned, real only | 18.3 % |
| **fine-tuned, + 5000 synth, real ×5** | **15.3 % ± 1.3** (best) |

So the recipe works; the open item is **throughput**, not accuracy.
