#!/usr/bin/env python3
"""Build a portable, self-contained data bundle for SLURM-cluster training.

Produces cluster/bundle/ with a flat, relative layout that works anywhere:

    bundle/
      images/<base>.jpg                 (all images, real + synth)
      page/<base>.xml                   (PAGE XML, imageFilename -> ../images/<base>.jpg)
      lists/recog_train.txt             (real train + synthetic)
      lists/recog_val.txt               (real held-out val)
      lists/seg.txt                     (all real, for segmentation training)

tar it (cluster/make_tarball.sh) and copy to the cluster; the SLURM scripts
run from bundle/ and resolve everything by relative path.
"""
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "cluster" / "bundle"
PAGE = OUT / "page"
IMG = OUT / "images"
LISTS = OUT / "lists"

REAL_TRAIN = REPO / "kraken_training" / "train_xml_combined.txt"
REAL_VAL = REPO / "kraken_training" / "val_xml_combined.txt"
SYNTH = REPO / "synthetic" / "synth_xml.txt"


def src_img(xml: Path) -> Path:
    s = str(xml)
    if "synthetic" in s:
        return REPO / "synthetic" / "images" / f"{xml.stem}.jpg"
    if "batch2_final" in s:
        return REPO / "batch2" / "images" / f"{xml.stem}.jpg"
    return REPO / "kraken_training" / "images" / f"{xml.stem}.jpg"


def read_list(p: Path):
    out = []
    for raw in p.read_text().splitlines():
        raw = raw.strip().replace("\\", "/")
        if raw:
            out.append(REPO / raw)
    return out


def stage(xmls):
    """Copy image + PAGE XML (imageFilename rewritten) into the bundle.
    Returns the list of bundle-relative page paths."""
    rels = []
    for xml in xmls:
        img = src_img(xml)
        if not (xml.exists() and img.exists()):
            continue
        base = xml.stem
        if not (IMG / f"{base}.jpg").exists():
            shutil.copy(img, IMG / f"{base}.jpg")
        if not (PAGE / f"{base}.xml").exists():
            text = re.sub(r'imageFilename="[^"]*"',
                          f'imageFilename="../images/{base}.jpg"',
                          xml.read_text(encoding="utf-8"), count=1)
            (PAGE / f"{base}.xml").write_text(text, encoding="utf-8")
        rels.append(f"page/{base}.xml")
    return rels


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in (PAGE, IMG, LISTS):
        d.mkdir(parents=True, exist_ok=True)

    real_train = read_list(REAL_TRAIN)
    real_val = read_list(REAL_VAL)
    synth = read_list(SYNTH)

    tr = stage(real_train) + stage(synth)        # recognition train = real + synth
    va = stage(real_val)                         # recognition val = real held-out
    seg = stage(real_train) + stage(real_val)    # segmentation = all real (already staged)

    tr_real = stage(real_train)                  # real-only (our best: Large+real)
    seg_all = seg + stage(synth)                 # segmentation on real + synth
    (LISTS / "recog_train.txt").write_text("\n".join(tr), encoding="utf-8")
    (LISTS / "recog_train_real.txt").write_text("\n".join(tr_real), encoding="utf-8")
    (LISTS / "recog_val.txt").write_text("\n".join(va), encoding="utf-8")
    (LISTS / "seg.txt").write_text("\n".join(seg), encoding="utf-8")
    (LISTS / "seg_all.txt").write_text("\n".join(seg_all), encoding="utf-8")

    n_img = len(list(IMG.glob("*.jpg")))
    print(f"bundle -> {OUT}")
    print(f"  images: {n_img}")
    print(f"  recog_train: {len(tr)}  recog_val: {len(va)}  seg: {len(seg)}")


if __name__ == "__main__":
    main()
