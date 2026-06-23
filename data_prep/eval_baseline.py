#!/usr/bin/env python3
"""Space-free letter-CER of ANY recognition model on a list of PAGE XML.
Run from inside the bundle dir:
    python ../eval_baseline.py <model.mlmodel> <val_xml_list.txt> [cpu|cuda:0]

Strips ALL whitespace from prediction AND ground truth before scoring, so an
off-the-shelf (space-emitting) model is compared fairly against our scriptio-
continua convention. Mirrors the kraken API used in synthetic/error_analysis.py.
"""
import os
import re
import sys
from pathlib import Path

from PIL import Image

# KEEP_DOTS=1 -> keep the middot · as a real character (strip only whitespace);
# default 0 -> letter-only (strip whitespace AND ·).
KEEP_DOTS = os.environ.get("KEEP_DOTS", "0") == "1"


def _norm(s: str) -> str:
    s = "".join(s.split())
    return s if KEEP_DOTS else s.replace("·", "")
from kraken.lib import models as kmodels
from kraken import rpred
from kraken.containers import Segmentation, BaselineLine

LINE_RE = re.compile(
    r'<TextLine[^>]*>\s*<Coords points="([^"]+)"[^>]*/>\s*'
    r'(?:<Baseline points="([^"]+)"[^>]*/>)?\s*<TextEquiv>\s*'
    r'<Unicode>([^<]*)</Unicode>', re.S)


def edit_distance(a: str, b: str) -> int:
    n = len(b)
    dp = list(range(n + 1))
    for i in range(1, len(a) + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def main() -> int:
    model_path, list_path = sys.argv[1], sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    model = kmodels.load_any(model_path, device=device)

    tot_e = tot_n = n_lines = n_pages = 0
    for rel in Path(list_path).read_text().split():
        rel = rel.strip()
        if not rel:
            continue
        xml = Path(rel)
        img = xml.parent.parent / "images" / (xml.stem + ".jpg")
        if not img.exists():
            print(f"  (missing image: {img})")
            continue
        im = Image.open(img).convert("L")
        W, H = im.size
        lines, gts = [], []
        for m in LINE_RE.finditer(xml.read_text(encoding="utf-8")):
            poly = [(min(max(int(x), 0), W - 1), min(max(int(y), 0), H - 1))
                    for x, y in (p.split(",") for p in m.group(1).split())]
            bl = ([(min(max(int(x), 0), W - 1), min(max(int(y), 0), H - 1))
                   for x, y in (p.split(",") for p in m.group(2).split())]
                  if m.group(2) else poly[:2])
            lines.append(BaselineLine(id=f"l{len(lines)}", baseline=bl, boundary=poly, text=m.group(3)))
            gts.append(m.group(3))
        if not lines:
            continue
        seg = Segmentation(type="baselines", imagename=str(img),
                           text_direction="horizontal-lr", script_detection=False, lines=lines)
        for rec, gt in zip(rpred.rpred(model, im, seg), gts):
            pred = rec.prediction if hasattr(rec, "prediction") else str(rec)
            p = _norm(pred)   # strip whitespace (+ · unless KEEP_DOTS)
            g = _norm(gt)
            if not g:
                continue
            tot_e += edit_distance(p, g)
            tot_n += len(g)
            n_lines += 1
        n_pages += 1

    cer = 100 * tot_e / max(1, tot_n)
    print(f"\nmodel: {model_path}")
    print(f"pages={n_pages} lines={n_lines} chars={tot_n}")
    print(f"space-free letter-CER = {cer:.2f}%   ({tot_e} edits / {tot_n} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
