#!/usr/bin/env python3
"""Build CV folds + synth lists for the 3-strategy comparison (all space-free).
Run once on the login node from inside the bundle dir:  python ../cv_folds.py

Outputs (all paths point to page_nospace/, scriptio continua):
  cv/fold{0..4}_train.txt / _val.txt   group-CV over the REAL inscriptions
  cv/synth_all.txt                     all synthetic (for 'mixed')
  cv/synth_train.txt / synth_val.txt   90/10 synth split (for 'pretrain')
"""
import os
import re
from pathlib import Path

# KEEP_DOTS=1 -> keep the middot · as a real character (strip only spaces) and
# write to page_dots/. Default 0 -> letter-only (strip spaces AND ·) -> page_nospace/.
KEEP_DOTS = os.environ.get("KEEP_DOTS", "0") == "1"
NSDIR = "page_dots" if KEEP_DOTS else "page_nospace"

PAGE = Path("page")
NS = Path(NSDIR)
CV = Path("cv")
K = 5
U = re.compile(r"(<Unicode>)([^<]*)(</Unicode>)")


def _clean(s):
    s = s.replace(" ", "")
    return s if KEEP_DOTS else s.replace("·", "")


def strip_to_ns(base):
    src = PAGE / f"{base}.xml"
    if not src.exists():
        return False
    t = U.sub(lambda m: m.group(1) + _clean(m.group(2)) + m.group(3),
              src.read_text(encoding="utf-8"))
    (NS / f"{base}.xml").write_text(t, encoding="utf-8")
    return True


def real_bases():
    out, seen = [], set()
    for lf in ("lists/recog_train_real.txt", "lists/recog_val.txt"):
        for rel in Path(lf).read_text().split():
            b = Path(rel.strip()).stem
            if rel.strip() and b not in seen:
                seen.add(b); out.append(b)
    return sorted(out)


def main():
    NS.mkdir(exist_ok=True)
    CV.mkdir(exist_ok=True)

    real = [b for b in real_bases() if strip_to_ns(b)]
    synth = sorted(p.stem for p in PAGE.glob("synth_*.xml"))
    synth = [b for b in synth if strip_to_ns(b)]

    # real group-CV — group by INSCRIPTION (HD number) so all photos of one
    # inscription stay in the same fold (no cross-photo text leakage).
    from collections import defaultdict
    groups = defaultdict(list)
    for b in real:
        groups[b.split("_")[0]].append(b)   # HD#### = inscription id
    hds = sorted(groups)
    folds = [[] for _ in range(K)]
    for i, hd in enumerate(hds):
        folds[i % K].extend(groups[hd])
    print(f"real pages={len(real)} inscriptions={len(hds)} "
          f"(dups: {sum(1 for v in groups.values() if len(v) > 1)} HD with >1 photo)")
    for k in range(K):
        val = folds[k]
        train = [b for j in range(K) if j != k for b in folds[j]]
        (CV / f"fold{k}_val.txt").write_text(
            "\n".join(f"{NSDIR}/{b}.xml" for b in val), encoding="utf-8")
        (CV / f"fold{k}_train.txt").write_text(
            "\n".join(f"{NSDIR}/{b}.xml" for b in train), encoding="utf-8")
        print(f"fold {k}: train={len(train)} val={len(val)}")

    # synth lists — ALL synth go into TRAINING (mixed / synthN). None is ever used
    # to validate the CV: every reported CER comes from cv/fold{k}_val.txt (real
    # held-out only). The 90/10 synth_train/synth_val split below is used ONLY by
    # the optional 'pretrain' strategy to early-stop the SYNTHETIC pre-training
    # phase — it never contributes to a reported number.
    (CV / "synth_all.txt").write_text(
        "\n".join(f"{NSDIR}/{b}.xml" for b in synth), encoding="utf-8")
    nval = max(1, len(synth) // 10)
    (CV / "synth_val.txt").write_text(
        "\n".join(f"{NSDIR}/{b}.xml" for b in synth[:nval]), encoding="utf-8")
    (CV / "synth_train.txt").write_text(
        "\n".join(f"{NSDIR}/{b}.xml" for b in synth[nval:]), encoding="utf-8")

    print(f"[{NSDIR}] real={len(real)}  synth={len(synth)} -> ALL synth into TRAIN; "
          f"CV val = real held-out only "
          f"(synth_train/val {len(synth)-nval}/{nval} exist for 'pretrain' only)")


if __name__ == "__main__":
    main()
