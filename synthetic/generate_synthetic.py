#!/usr/bin/env python3
"""
generate_synthetic.py

Renders synthetic Latin-inscription training images + PAGE XML from real
EDH transcriptions, per SYNTHETIC_PLAN.md.

Pipeline (8 stages):
  1. TEXT    sample inscription lines from edh_inscriptions.json
             (bracket-free, 2-40 chars, >=2 lines, Latin charset),
             stratified by province (proportional, capped 25%)
  2. LAYOUT  alignment (centered/left), diminishing line heights,
             per-line baseline rotation + sinusoidal sag, crowding
  3. GLYPHS  weighted font mix (Capitalis-like OFL fonts), per-glyph
             scale jitter + shear ("schraeg"), optional condensing (S2),
             interpuncts between words (p~0.6)
  4. RELIEF  carved look: text mask -> blurred height map -> directional
             lighting (bright/dark opposing edges) + darkened groove
  5. SURFACE procedural stone (limestone/marble/sandstone, seeded
             fractal noise -- no downloaded textures, no license issues)
  6. DAMAGE  erosion mask eating glyph edges, cracks, pits
  7. CAMERA  global affine (rotation/shear -- coordinates transformed
             exactly), brightness gradient, vignette, sensor noise,
             blur, random JPEG quality
  8. OUTPUT  synthetic/images/synth_NNNNN.jpg
             synthetic/page/synth_NNNNN.xml   (exact baselines + masks)
             synthetic/review.html            (contact sheet for QA)
             synthetic/synth_xml.txt          (list for ketos -t)

GT mapping: BOTH image and GT are stone-form (upper, U->V, J->I), matching
  the real data's page_final convention. Interpuncts rendered but only in GT
  if the EDH line contains them.

Usage:
    python synthetic/generate_synthetic.py --n 500 --seed 42
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent          # synthetic/
REPO = ROOT.parent
PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
ET.register_namespace("", PAGE_NS)

# Layout knobs, overridable by experimental variants (generate_experimental.py).
# Defaults reproduce the stable v3.1 behaviour exactly.
PARAMS = {
    "margin_x": (0.07, 0.13),     # horizontal margin as fraction of W
    "gap": (0.35, 0.75),          # inter-line gap vs line height
    "unit_clamp": (26, 60),       # absolute line-height clamp in px
    "rot_jitter_deg": 0.0,        # per-glyph rotation jitter (0 = off)
}

# ---------------------------------------------------------------------------
# Stage 1 — TEXT
# ---------------------------------------------------------------------------
# letters, space, interpunct, centuria-symbol | — NO brackets: Leiden
# bracket notation was ruled an annotation error and removed from the
# real GT as well (user decision), so synth never generates it either.
LINE_RE = re.compile(r"^[A-Za-z·| ]{2,45}$")


def valid_line(ln: str) -> bool:
    return LINE_RE.match(ln) is not None


def tokenize_word(word: str) -> list[tuple[str, str, bool]]:
    """Split a GT word into (render_str, gt_str, lost) tokens.

    '[t]' becomes one atomic token: render 'T' as a DESTROYED patch
    (lost=True), GT keeps '[t]'. Atomic so fragmentation can never
    produce unbalanced brackets.
    """
    toks: list[tuple[str, str, bool]] = []
    i = 0
    while i < len(word):
        if word[i] == "[":
            j = word.index("]", i)
            inner = word[i + 1:j]
            toks.append((to_stone_form(inner), word[i:j + 1], True))
            i = j + 1
        else:
            toks.append((to_stone_form(word[i]), word[i], False))
            i += 1
    return toks


def load_corpus(json_path: Path, rng: random.Random,
                max_share: float = 0.25,
                no_photos_only: bool = False,
                exclude_hds: frozenset = frozenset()) -> list[dict]:
    """Load EDH, keep inscriptions whose diplomatic lines all pass the
    v1 filter, return a province-stratified shuffled pool.

    no_photos_only: keep ONLY inscriptions without photos (has_photos False)
    exclude_hds:    drop these HD numbers (e.g. our real annotated set)
    -> guarantees no text/HD overlap with any photographed/real data (no leakage)."""
    print(f"Loading {json_path} ...")
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    pool: list[dict] = []
    sk_photo = sk_excl = 0
    for insc in data["inscriptions"]:
        if no_photos_only and insc.get("has_photos"):
            sk_photo += 1; continue
        if str(insc.get("hd_nr", "")) in exclude_hds:
            sk_excl += 1; continue
        lines = [(ln.get("diplomatic") or "").strip() for ln in insc.get("lines") or []]
        lines = [ln for ln in lines if ln]
        if not (2 <= len(lines) <= 12):       # real p90 is 12 lines/stone
            continue
        if not all(valid_line(ln) for ln in lines):
            continue
        n_br = sum(ln.count("[") for ln in lines)
        n_ch = sum(len(ln) for ln in lines)
        if n_br * 4 > n_ch:                   # mostly-restored texts are useless
            continue
        pool.append({
            "hd_nr": insc.get("hd_nr", "?"),
            "province": (insc.get("province") or "unknown").strip() or "unknown",
            "lines": lines,
            "n_chars": n_ch,
            "has_br": n_br > 0,
        })

    # Proportional-capped province stratification: shuffle within province,
    # cap any single province at max_share of the pool we draw from.
    # Degenerate case: if province metadata is missing for (almost) all
    # records (the EDH fetcher currently doesn't extract it), capping would
    # just throw away 75% of usable data — skip it then.
    print(f"  filtered: {len(pool)} usable  (skipped {sk_photo} with-photo, {sk_excl} excluded-HD)")
    by_prov: dict[str, list[dict]] = {}
    for item in pool:
        by_prov.setdefault(item["province"], []).append(item)
    for items in by_prov.values():
        rng.shuffle(items)
    def weighted_order(items: list[dict]) -> list[dict]:
        # Efraimidis-Spirakis weighted sampling without replacement: bias the
        # draw order toward text-rich inscriptions — our real annotated set
        # skews to substantial stones (median 6 lines / 13 chars per line).
        return sorted(items,
                      key=lambda it: rng.random() ** (1.0 / max(1, it["n_chars"])),
                      reverse=True)

    if len(by_prov) <= 2:
        capped = weighted_order(pool)
        print(f"  {len(pool)} usable inscriptions "
              f"(province metadata absent -> stratification skipped)")
        return capped
    cap = int(len(pool) * max_share)
    capped = []
    for prov, items in by_prov.items():
        capped.extend(items[:cap] if cap > 0 else items)
    capped = weighted_order(capped)
    print(f"  {len(pool)} usable inscriptions -> {len(capped)} after province cap "
          f"({len(by_prov)} provinces)")
    return capped


def to_stone_form(gt: str) -> str:
    """GT (EDH diplomatic, mixed case) -> what the stone shows."""
    s = gt.upper().replace("U", "V").replace("J", "I")
    return s


# ---------------------------------------------------------------------------
# Stage 5 (prepared early; needed as canvas) — SURFACE: procedural stone
# ---------------------------------------------------------------------------
def fractal_noise(h: int, w: int, rng: np.random.Generator,
                  octaves: int = 5, persistence: float = 0.55) -> np.ndarray:
    """Cheap fractal value noise in [0,1] via upsampled random grids."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gh = max(2, h // (2 ** (octaves - o)))
        gw = max(2, w // (2 ** (octaves - o)))
        grid = rng.random((gh, gw)).astype(np.float32)
        img = Image.fromarray((grid * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        out += amp * (np.asarray(img, np.float32) / 255.0)
        total += amp
        amp *= persistence
    return out / total


STONE_PALETTES = {
    # weighted per our annotated set: limestone ~ marble > sandstone
    "limestone": {"base": (201, 193, 172), "w": 0.40},
    "marble":    {"base": (216, 213, 207), "w": 0.35},
    "sandstone": {"base": (188, 158, 118), "w": 0.25},
}


def make_stone(w: int, h: int, kind: str, nrng: np.random.Generator) -> np.ndarray:
    """Procedural stone texture as float RGB array (0..255)."""
    base = np.array(STONE_PALETTES[kind]["base"], np.float32)
    n = fractal_noise(h, w, nrng, octaves=6)
    grain = fractal_noise(h, w, nrng, octaves=3)
    tex = np.ones((h, w, 3), np.float32) * base[None, None, :]
    tex += (n[..., None] - 0.5) * 38.0          # large-scale tonal variation
    tex += (grain[..., None] - 0.5) * 14.0      # fine grain

    if kind == "marble":
        xx = np.linspace(0, 1, w)[None, :].repeat(h, 0)
        turb = fractal_noise(h, w, nrng, octaves=5)
        veins = np.abs(np.sin((xx * nrng.uniform(2, 5) + turb * nrng.uniform(2.5, 5.0)) * math.pi))
        veins = (1.0 - veins) ** 6              # thin dark veins
        tex -= veins[..., None] * nrng.uniform(25, 60)
    elif kind == "sandstone":
        yy = np.linspace(0, 1, h)[:, None].repeat(w, 1)
        band = np.sin(yy * nrng.uniform(20, 50) * math.pi + fractal_noise(h, w, nrng, 3) * 4)
        tex += band[..., None] * 6.0            # horizontal bedding

    # low-frequency lighting gradient across the slab
    gx, gy = nrng.uniform(-1, 1), nrng.uniform(-1, 1)
    xx = np.linspace(-0.5, 0.5, w)[None, :]
    yy = np.linspace(-0.5, 0.5, h)[:, None]
    tex += ((gx * xx + gy * yy) * nrng.uniform(10, 45))[..., None]
    return np.clip(tex, 0, 255)


_REAL_BGS: list[Path] | None = None


def load_real_background(W: int, H: int, rng: random.Random) -> np.ndarray | None:
    """Random harvested patch from our real photos, flipped/rotated/stretched.

    Returns None if synthetic/real_backgrounds/ is empty (harvester not run).
    """
    global _REAL_BGS
    if _REAL_BGS is None:
        _REAL_BGS = sorted((ROOT / "real_backgrounds").glob("*.jpg"))
    if not _REAL_BGS:
        return None
    # rejection-sample a usable patch. Criteria calibrated on all 264
    # harvested patches: structure score (mean gradient magnitude) median
    # is 9.0, ornament/molding crops score 16-26 -> cutoff 13 (p75)
    # rejects exactly the busy decorated surfaces; luminance bounds drop
    # deep-shadow crops.
    img = None
    for _ in range(8):
        cand = Image.open(rng.choice(_REAL_BGS)).convert("RGB")
        a = np.asarray(cand.convert("L"), np.float32)
        gy, gx = np.gradient(a)
        structure = float(np.hypot(gx, gy).mean())
        if 70 <= a.mean() <= 235 and structure <= 13.0:
            img = cand
            # brightness lift for darkish stone so carved text stays legible
            if a.mean() < 115:
                arr = np.asarray(img, np.float32) * min(1.8, 130.0 / a.mean())
                img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            break
    if img is None:
        return None                       # fall back to procedural stone
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    rot = rng.choice([0, 90, 180, 270])
    if rot:
        img = img.rotate(rot, expand=True)
    # mirror-tile at near-native scale instead of stretching: upscaling a
    # 220-420px patch to 900px smoothed away the high-frequency texture
    # (measured as too-low strip contrast). Tiles are kept large enough
    # that at most ~1 mirror seam per axis is visible (kaleidoscope guard).
    scale = max(rng.uniform(0.9, 1.4),
                0.65 * W / img.width, 0.65 * H / img.height)
    tw, th = max(64, int(img.width * scale)), max(64, int(img.height * scale))
    tile = img.resize((tw, th), Image.LANCZOS)
    tile_fl = tile.transpose(Image.FLIP_LEFT_RIGHT)
    canvas = Image.new("RGB", (W, H))
    y = 0
    row = 0
    while y < H:
        x, col = 0, 0
        while x < W:
            t = tile if (row + col) % 2 == 0 else tile_fl
            if row % 2 == 1:
                t = t.transpose(Image.FLIP_TOP_BOTTOM)
            canvas.paste(t, (x, y))
            x += tw
            col += 1
        y += th
        row += 1
    return np.asarray(canvas, np.float32)


# ---------------------------------------------------------------------------
# Fragmentation — broken/torn slabs (common in EDH; GT-consistent by design)
# ---------------------------------------------------------------------------
def make_fragment_mask(W: int, H: int, rng: random.Random,
                       nrng: np.random.Generator) -> np.ndarray | None:
    """1.0 = surviving stone, 0.0 = lost. 1-3 ragged half-plane 'bites'.

    Returns None if the surviving area would drop below 45% (bite skipped),
    leaving at least the slab's core intact.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    frag = np.ones((H, W), np.float32)
    n_bites = rng.randint(1, 3)
    for _ in range(n_bites):
        # anchor the cut near a border so the centre tends to survive
        edge = rng.choice(["l", "r", "t", "b"])
        if edge == "l":   ax, ay = 0, rng.uniform(0, H)
        elif edge == "r": ax, ay = W, rng.uniform(0, H)
        elif edge == "t": ax, ay = rng.uniform(0, W), 0
        else:             ax, ay = rng.uniform(0, W), H
        theta = math.atan2(H / 2 - ay, W / 2 - ax) + rng.uniform(-0.7, 0.7)
        depth = rng.uniform(0.12, 0.38) * min(W, H)   # how far the bite reaches
        s = (np.cos(theta) * (xx - ax) + np.sin(theta) * (yy - ay))
        ragged = (fractal_noise(H, W, nrng, octaves=4) - 0.5) * rng.uniform(40, 110)
        bite = (s + ragged) < depth
        cand = frag * (~bite).astype(np.float32)
        if cand.mean() >= 0.45:
            frag = cand
    return frag if frag.mean() < 0.999 else None


def apply_fracture(arr: np.ndarray, frag: np.ndarray,
                   nrng: np.random.Generator) -> np.ndarray:
    """Composite the broken slab over a backdrop with fracture-edge effects."""
    h, w = frag.shape
    # backdrop: dark museum cloth or plaster wall
    if nrng.random() < 0.55:
        bg_col = np.array([nrng.uniform(25, 70)] * 3, np.float32) \
                 + nrng.uniform(-6, 6, 3).astype(np.float32)
    else:
        base = nrng.uniform(140, 185)
        bg_col = np.array([base, base * nrng.uniform(0.95, 1.0),
                           base * nrng.uniform(0.88, 0.98)], np.float32)
    bg = np.ones_like(arr) * bg_col[None, None, :]
    bg += (fractal_noise(h, w, nrng, 4)[..., None] - 0.5) * 18

    fimg = Image.fromarray((frag * 255).astype(np.uint8))
    eroded = np.asarray(fimg.filter(ImageFilter.MinFilter(9)), np.float32) / 255.0
    dilated = np.asarray(fimg.filter(ImageFilter.MaxFilter(15)), np.float32) / 255.0
    edge_band = np.clip(frag - eroded, 0, 1)        # fresh fracture face
    shadow_band = np.clip(dilated - frag, 0, 1)     # contact shadow on backdrop

    out = arr * frag[..., None] + bg * (1 - frag[..., None])
    out += edge_band[..., None] * nrng.uniform(10, 45)      # bright break face
    out -= (shadow_band * nrng.uniform(20, 60))[..., None]  # drop shadow
    return np.clip(out, 0, 255)


# ---------------------------------------------------------------------------
# Stages 2+3 — LAYOUT + GLYPHS  (renders the text mask, returns geometry)
# ---------------------------------------------------------------------------
@dataclass
class LineGeom:
    baseline: list[tuple[float, float]]   # polyline, left->right
    top: float                            # mask top y at line start
    bottom: float                         # mask bottom y
    x0: float
    x1: float
    gt: str


FONT_FILES = [
    # S1 Capitalis monumentalis
    ("Cinzel-Variable.ttf", 3.0),
    ("Marcellus-Regular.ttf", 2.0),
    ("Forum-Regular.ttf", 2.0),
    ("Cardo-Regular.ttf", 1.0),
    ("Cardo-Bold.ttf", 1.0),
    # S2-leaning (rougher)
    ("Caudex-Regular.ttf", 1.0),
    ("Caudex-Bold.ttf", 1.0),
    # S4 late-antique / early-Christian (uncial influence, crude carving)
    ("UncialAntiqua-Regular.ttf", 0.6),
    ("MedievalSharp.ttf", 0.4),
    ("AlmendraSC-Regular.ttf", 0.5),
]


def pick_font(rng: random.Random) -> Path:
    names, weights = zip(*FONT_FILES)
    return ROOT / "fonts" / rng.choices(names, weights=weights, k=1)[0]


_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def get_font(path: Path, px: int) -> ImageFont.FreeTypeFont:
    key = (str(path), px)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(str(path), px)
    return _FONT_CACHE[key]


def render_text_mask(lines_gt: list[str], W: int, H: int, rng: random.Random,
                     font_path: Path, frag: np.ndarray | None = None
                     ) -> tuple[np.ndarray, list[LineGeom]]:
    """Render all lines as a grayscale mask (0..1) + exact geometry.

    One font per inscription (a stone is carved by one hand), but
    per-glyph jitter within it. Implements: alignment, diminishing
    heights, baseline rotation+sag, condensing (S2), crowding,
    interpuncts, per-glyph shear/scale.
    """
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    damage = Image.new("L", (W, H), 0)     # destroyed patches ([x] supplied)
    ddraw = ImageDraw.Draw(damage)
    n = len(lines_gt)

    # --- vertical layout: diminishing heights ---------------------------
    margin_y = H * rng.uniform(0.06, 0.14)
    usable_h = H - 2 * margin_y
    shrink = rng.uniform(0.55, 1.0)               # height factor line1 -> lineN
    rel = [1.0 + (shrink - 1.0) * (i / max(1, n - 1)) for i in range(n)]
    gap = rng.uniform(*PARAMS["gap"])            # inter-line gap vs line height
    total = sum(rel) * (1 + gap) - gap * rel[-1]
    unit = usable_h / total                       # height of a rel=1.0 line
    # ABSOLUTE clamp targeting the real strip-height distribution: real line
    # strips are 30-86 px (median 52) on ~800px-wide photos; v1 synth rendered
    # 113 px medians -> systematic scale mismatch after Kraken's fixed-height
    # line normalization. Text no longer fills the slab — real steles don't
    # fill theirs either.
    unit = min(unit, rng.uniform(*PARAMS["unit_clamp"]))
    # random vertical placement of the text block in the leftover space
    leftover = max(0.0, usable_h - unit * total)
    y_offset = rng.uniform(0.0, 0.7) * leftover

    align = rng.choices(["center", "left"], weights=[0.65, 0.35], k=1)[0]
    margin_x = W * rng.uniform(*PARAMS["margin_x"])  # default >=7%: rotation-safe
    condense = rng.uniform(0.72, 1.0) if rng.random() < 0.35 else 1.0   # S2
    inscription_shear = rng.uniform(-0.10, 0.10)  # carver's habitual slant
    geoms: list[LineGeom] = []
    y_cursor = margin_y + y_offset

    for i, gt in enumerate(lines_gt):
        line_h = unit * rel[i]
        font_px = max(14, int(line_h))
        font = get_font(font_path, font_px)
        word_tokens = [tokenize_word(w) for w in gt.split(" ")]
        stone_words = ["".join(t[0] for t in toks) for toks in word_tokens]
        n_render_chars = sum(len(sw) for sw in stone_words)
        render_interpunct = rng.random() < 0.6

        # measure natural width (approx, before jitter)
        tracking = rng.uniform(0.00, 0.10) * font_px
        nat = sum(font.getlength(sw) for sw in stone_words)
        sp = font.getlength(" ") * (0.6 if render_interpunct else 1.0)
        nat += sp * (len(stone_words) - 1) + tracking * max(0, n_render_chars - 1)
        # shrink font if line would overflow (0.93 = jitter/rotation headroom)
        avail = (W - 2 * margin_x) * 0.93
        if nat > avail:
            scale = avail / nat
            font_px = max(12, int(font_px * scale))
            font = get_font(font_path, font_px)
            nat *= scale
        x_start = (W - nat) / 2 if align == "center" else margin_x

        # baseline shape: rotation + sinusoidal sag
        slope = math.tan(math.radians(rng.uniform(-3.0, 3.0)))
        sag_amp = rng.uniform(0, 0.06) * font_px
        sag_lambda = rng.uniform(0.7, 1.6) * W
        sag_phase = rng.uniform(0, 2 * math.pi)
        baseline_y0 = y_cursor + font_px           # pen baseline at line start

        def base_y(x: float) -> float:
            return (baseline_y0 + slope * (x - x_start)
                    + sag_amp * math.sin(2 * math.pi * x / sag_lambda + sag_phase))

        # ---- glyph-by-glyph rendering --------------------------------
        ascent, descent = font.getmetrics()
        pen_x = x_start
        # vertical wobble: smooth random walk (carvers drift, they don't jump)
        wobble = 0.0
        wobble_step = rng.uniform(0.008, 0.032) * font_px   # per-stone character
        wobble_max = 0.035 * font_px
        kept_words: list[list[str]] = [[] for _ in word_tokens]
        kept_x0, kept_x1 = None, None

        def survives(cx: float, cy: float) -> bool:
            """Is this point on surviving stone? (GT-consistency gate)"""
            if frag is None:
                return True
            xi = min(max(int(cx), 0), W - 1)
            yi = min(max(int(cy), 0), H - 1)
            return frag[yi, xi] >= 0.5

        for wi, toks in enumerate(word_tokens):
            for ti, (render, gt_piece, lost) in enumerate(toks):
                # crowding: tighten tracking over the last 30% of the line
                progress = (pen_x - x_start) / max(1.0, nat)
                crowd = 1.0 - 0.25 * max(0.0, (progress - 0.7) / 0.3)
                tok_adv = sum(font.getlength(c) for c in render) * condense \
                    + tracking * len(render)
                gy = base_y(pen_x) + wobble

                if lost:
                    # supplied restoration [x]: the letters are physically
                    # destroyed — render a damage patch instead of glyphs,
                    # keep '[x]' in the GT (the convention the model must emit)
                    if survives(pen_x + tok_adv / 2, gy - font_px * 0.35):
                        pad = font_px * rng.uniform(0.10, 0.30)
                        ddraw.ellipse([pen_x - pad, gy - ascent - pad,
                                       pen_x + tok_adv + pad, gy + descent * 0.3 + pad],
                                      fill=int(rng.uniform(140, 255)))
                        kept_words[wi].append(gt_piece)
                        kept_x0 = pen_x if kept_x0 is None else min(kept_x0, pen_x)
                        kept_x1 = max(kept_x1 or pen_x, pen_x + tok_adv)
                    pen_x += tok_adv * crowd
                    continue

                ch = render                      # normal token = single glyph
                wobble = max(-wobble_max,
                             min(wobble_max, wobble + rng.uniform(-1, 1) * wobble_step))
                adv = font.getlength(ch) * condense
                # survival test at glyph centre: a glyph lost to the break is
                # neither rendered nor part of the GT (the pen still advances
                # -- surviving letters keep their stone positions)
                if survives(pen_x + adv / 2, gy - font_px * 0.35):
                    gsc = rng.uniform(0.92, 1.08)                   # size jitter
                    gshear = inscription_shear + rng.uniform(-0.05, 0.05)
                    gpx = max(8, int(font_px * gsc))
                    gfont = get_font(font_path, gpx)
                    gl = gfont.getlength(ch)
                    tile_w = int(gl + abs(gshear) * gpx * 2 + 8)
                    tile_h = int(gpx * 1.6 + 8)
                    tile = Image.new("L", (tile_w, tile_h), 0)
                    ImageDraw.Draw(tile).text((4, 4), ch, font=gfont, fill=255)
                    if abs(gshear) > 0.005:
                        tile = tile.transform(
                            tile.size, Image.AFFINE,
                            (1, gshear, -gshear * tile_h / 2, 0, 1, 0),
                            resample=Image.BILINEAR)
                    if PARAMS["rot_jitter_deg"] > 0:
                        tile = tile.rotate(
                            rng.uniform(-PARAMS["rot_jitter_deg"],
                                        PARAMS["rot_jitter_deg"]),
                            expand=True, resample=Image.BILINEAR)
                    gasc, _ = gfont.getmetrics()
                    mask.paste(tile, (int(pen_x) - 4, int(gy - gasc) - 4), tile)
                    kept_words[wi].append(gt_piece)
                    kept_x0 = pen_x if kept_x0 is None else min(kept_x0, pen_x)
                    kept_x1 = pen_x + adv if kept_x1 is None else max(kept_x1, pen_x + adv)
                # pseudo-ligature (nexus approximation): occasionally pull the
                # next in-word glyph into overlap. True fused-stroke ligatures
                # remain open for v2 (glyph-stamp composition).
                lig = rng.uniform(0.55, 0.75) if (ti < len(toks) - 1
                                                  and rng.random() < 0.04) else 1.0
                pen_x += adv * crowd * lig + tracking
            if wi < len(word_tokens) - 1:
                # word gap, optionally with interpunct (drawn as carved dot)
                gap_w = sp
                if render_interpunct:
                    cx = pen_x + gap_w / 2
                    cy = base_y(cx) - font_px * 0.32
                    if survives(cx, cy):
                        r = max(1.5, font_px * 0.055)
                        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
                pen_x += gap_w

        # GT after fragmentation: surviving chars only, empty words dropped.
        # STONE-FORM (upper, U->V, J->I) so the label matches BOTH the rendered
        # stone-form image AND the real data's stone-form GT (page_final).
        # Bug fixed 2026-06: synth GT was kept in lowercase EDH form while the
        # image rendered stone-form -> the pretrain learned image=AVRELIVS ->
        # label=aurelius, the WRONG mapping for the real finetune, so synth
        # pretraining washed out (pretrain ~= no-pretrain).
        kept_gt = to_stone_form(" ".join("".join(wchars) for wchars in kept_words if wchars))
        n_kept = sum(len(wchars) for wchars in kept_words)
        if n_kept >= 2 and kept_x0 is not None:
            # geometry (6-point baseline polyline over the SURVIVING span)
            xs = np.linspace(kept_x0, max(kept_x1, kept_x0 + 1), 6)
            bl = [(float(x), float(base_y(x))) for x in xs]
            top = min(base_y(x) for x in xs) - ascent * 1.02
            bottom = max(base_y(x) for x in xs) + descent * 0.4
            geoms.append(LineGeom(baseline=bl, top=top, bottom=bottom,
                                  x0=kept_x0, x1=kept_x1, gt=kept_gt))
        # advance: never less than 1.5x the rendered font height, so
        # cap-height + descent + wobble/sag of adjacent lines cannot
        # collide even when gap is small
        y_cursor += max(line_h * (1 + gap), font_px * 1.5)

    return (np.asarray(mask, np.float32) / 255.0, geoms,
            np.asarray(damage, np.float32) / 255.0)


# ---------------------------------------------------------------------------
# Stage 4 — RELIEF (carved lighting)   Stage 6 — DAMAGE
# ---------------------------------------------------------------------------
def structural_erosion(mask: np.ndarray, font_px: float,
                       nrng: np.random.Generator) -> np.ndarray:
    """Preferentially erode THIN HORIZONTAL strokes (crossbars, arms, serifs)
    while sparing vertical stems and round bowls.

    This targets the model's dominant real failure mode: on weathered stone the
    thin shallow crossbars of T/E/F/L erode first, leaving the vertical stem ->
    the letter reads as 'I'. Isotropic edge-erosion never reproduces this, so
    the synth was redundant with the clean CATMuS base. Here a crossbar can be
    thinned or fully lost while the GT label stays T/E/F/L, teaching the model
    the worn-stub <-> capital ambiguity with a known target.

    Returns a multiplicative keep-map in [0,1] over the glyph mask.
    """
    mb = mask > 0.4
    if not mb.any():
        return np.ones_like(mask)
    vse = np.ones((max(4, int(font_px * 0.40)), 1), bool)   # tall: finds stems
    hse = np.ones((1, max(4, int(font_px * 0.22))), bool)   # wide: finds bars
    stems = ndi.binary_opening(mb, structure=vse)
    bars = ndi.binary_opening(mb, structure=hse)
    # thin horizontal = wide run that is NOT part of a vertical stem
    thin = bars & ~ndi.binary_dilation(stems, iterations=max(1, int(font_px * 0.06)))
    if not thin.any():
        return np.ones_like(mask)
    # soften the thin region so partial loss looks natural, then carve it with
    # fractal noise. ~1 in 4 inscriptions gets heavy near-total crossbar loss.
    heavy = nrng.random() < 0.28
    strength = nrng.uniform(0.85, 1.0) if heavy else nrng.uniform(0.35, 0.75)
    thr = nrng.uniform(0.30, 0.45) if heavy else nrng.uniform(0.45, 0.62)
    h, w = mask.shape
    noise = fractal_noise(h, w, nrng, octaves=5)
    thin_soft = np.asarray(Image.fromarray((thin * 255).astype(np.uint8))
                           .filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255.0
    eat = strength * (noise > thr).astype(np.float32) * thin_soft
    return np.clip(1.0 - eat, 0.0, 1.0)


def carve(stone: np.ndarray, mask: np.ndarray, nrng: np.random.Generator,
          damage: np.ndarray | None = None, font_px: float = 90.0) -> np.ndarray:
    """Composite the text mask onto stone as a carved (v-cut) relief.

    font_px scales the relief blur: at small letter sizes the strokes are
    only 3-6 px wide and a fixed-radius blur flattens the height-map
    gradients entirely (measured: strip contrast stuck at 18 vs real 30).
    """
    h, w = mask.shape
    if damage is not None and damage.max() > 0:
        # destroyed patches where supplied [x] letters once stood: battered,
        # mottled surface — visibly damaged, no letterforms left
        dmg = np.asarray(Image.fromarray((damage * 255).astype(np.uint8))
                         .filter(ImageFilter.GaussianBlur(4)), np.float32) / 255.0
        rough = fractal_noise(h, w, nrng, octaves=6)
        stone = stone - (dmg * (15 + 70 * rough))[..., None]
        stone = np.clip(stone, 0, 255)
    # 6. DAMAGE first: erosion eats the carving before lighting is computed
    erosion = fractal_noise(h, w, nrng, octaves=5)
    erosion_strength = nrng.uniform(0.0, 0.45)   # >0.55 ate strokes entirely
    keep = 1.0 - erosion_strength * (erosion > nrng.uniform(0.55, 0.75)).astype(np.float32)
    # structural erosion: preferentially destroy thin horizontal strokes
    # (crossbars/arms/serifs) so T/E/F/L can wear down toward an I-stub while
    # keeping the GT label — teaches the dominant real confusion (see analysis).
    struct_keep = structural_erosion(mask, font_px, nrng)
    m = mask * keep * struct_keep

    # height map of the groove
    depth_img = Image.fromarray((m * 255).astype(np.uint8))
    blur_r = nrng.uniform(0.8, 3.2) * max(0.35, min(1.0, font_px / 90.0))
    height = np.asarray(depth_img.filter(ImageFilter.GaussianBlur(blur_r)),
                        np.float32) / 255.0
    gy, gx = np.gradient(height)

    az = nrng.uniform(0, 2 * math.pi)             # light azimuth
    # contrast floor raised after visual review + dashboard (synth strip
    # contrast was median 19.8 vs real 30.5, p10 nearly invisible at 10.2)
    strength = nrng.uniform(280, 540)
    shade = (gx * math.cos(az) + gy * math.sin(az)) * strength
    if nrng.random() < 0.15:
        # raised relief (litterae caelatae / cast bronze style): letters catch
        # the light instead of sitting in a shadowed groove
        shade = -shade
        groove_dark = -m * nrng.uniform(20, 60)   # letters brightened
    else:
        groove_dark = m * nrng.uniform(38, 95)    # groove floor in shadow

    # modulate carving contrast by LOCAL surface brightness so letters in
    # shadowed photo regions don't glow ("pasted-on" artifact on real bgs)
    local_lum = np.asarray(
        Image.fromarray(stone.mean(axis=2).astype(np.uint8))
             .filter(ImageFilter.GaussianBlur(25)), np.float32) / 255.0
    lum_mod = 0.55 + 0.45 * local_lum
    out = stone + (shade * lum_mod)[..., None] - (groove_dark * lum_mod)[..., None]

    # cracks: a few dark random-walk polylines with a bright offset edge
    crack_layer = Image.new("L", (w, h), 0)
    cd = ImageDraw.Draw(crack_layer)
    for _ in range(int(nrng.integers(0, 4))):
        x, y = nrng.uniform(0, w), nrng.uniform(0, h)
        ang = nrng.uniform(0, 2 * math.pi)
        pts = [(x, y)]
        for _ in range(int(nrng.integers(10, 40))):
            ang += nrng.uniform(-0.6, 0.6)
            x += math.cos(ang) * nrng.uniform(5, 18)
            y += math.sin(ang) * nrng.uniform(5, 18)
            pts.append((x, y))
        cd.line(pts, fill=255, width=int(nrng.integers(1, 3)))
    crack = np.asarray(crack_layer, np.float32) / 255.0
    out -= crack[..., None] * nrng.uniform(25, 70)
    out += np.roll(crack, 2, axis=1)[..., None] * nrng.uniform(5, 20)

    # pits
    pit = fractal_noise(h, w, nrng, octaves=6)
    out -= ((pit > 0.78).astype(np.float32) * nrng.uniform(8, 30))[..., None]
    return np.clip(out, 0, 255)


# ---------------------------------------------------------------------------
# Stage 7 — CAMERA (global affine + photometric)
# ---------------------------------------------------------------------------
def camera(img_arr: np.ndarray, geoms: list[LineGeom],
           rng: random.Random, nrng: np.random.Generator):
    """Returns (image, transformed_geoms, jpeg_quality, fwd_transform)."""
    h, w = img_arr.shape[:2]
    img = Image.fromarray(img_arr.astype(np.uint8))

    # global affine: rotation + slight shear, exact coordinate transform
    rot = math.radians(rng.uniform(-3.0, 3.0))
    shx = rng.uniform(-0.04, 0.04)
    cx, cy = w / 2, h / 2
    cos, sin = math.cos(rot), math.sin(rot)
    # forward map (x,y)->(x',y'): translate to center, shear, rotate, back
    def fwd(x: float, y: float) -> tuple[float, float]:
        x, y = x - cx, y - cy
        x = x + shx * y                       # shear
        x, y = x * cos - y * sin, x * sin + y * cos
        return x + cx, y + cy
    # PIL wants the inverse map coefficients (output -> input)
    def inv(xp: float, yp: float) -> tuple[float, float]:
        x, y = xp - cx, yp - cy
        x, y = x * cos + y * sin, -x * sin + y * cos   # un-rotate
        x = x - shx * y                                # un-shear
        return x + cx, y + cy
    # build affine coeff numerically from inv() at basis points
    p00, p10, p01 = inv(0, 0), inv(1, 0), inv(0, 1)
    coeffs = (p10[0] - p00[0], p01[0] - p00[0], p00[0],
              p10[1] - p00[1], p01[1] - p00[1], p00[1])
    img = img.transform((w, h), Image.AFFINE, coeffs,
                        resample=Image.BILINEAR, fillcolor=(120, 115, 105))

    new_geoms = []
    for g in geoms:
        bl = [fwd(x, y) for x, y in g.baseline]
        # carry ascender/descender as offsets relative to the (new) baseline
        rel_asc = g.baseline[0][1] - g.top
        rel_desc = g.bottom - g.baseline[0][1]
        new_geoms.append(LineGeom(
            baseline=bl,
            top=bl[0][1] - rel_asc, bottom=bl[0][1] + rel_desc,
            x0=g.x0, x1=g.x1, gt=g.gt))

    arr = np.asarray(img, np.float32)
    # vignette
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    arr *= (1.0 - nrng.uniform(0.05, 0.30) * d ** 2)[..., None]
    # exposure / white balance (widened)
    arr *= nrng.uniform(0.70, 1.25)
    arr[..., 0] *= nrng.uniform(0.95, 1.06)
    arr[..., 2] *= nrng.uniform(0.94, 1.05)

    # B/W archive-photo simulation (a large share of EDH photos are old
    # monochrome archive shots — closes the most visible photometric gap)
    if rng.random() < 0.40:
        gray = arr.mean(axis=2, keepdims=True)
        # harsh tone curve typical of archive film
        gamma = nrng.uniform(0.65, 1.5)
        gray = np.clip(gray / 255.0, 0, 1) ** gamma * 255.0
        mode = rng.random()
        if mode < 0.65:
            arr = np.repeat(gray, 3, axis=2)                    # pure B/W
        else:
            sepia = np.array([1.12, 0.97, 0.78], np.float32)    # aged print
            arr = gray * sepia[None, None, :]
        arr += nrng.normal(0, nrng.uniform(2.0, 7.0), arr.shape)  # film grain

    # sensor noise
    arr += nrng.normal(0, nrng.uniform(1.0, 5.0), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    # blur (cap reduced for the small v3 letter sizes)
    if rng.random() < 0.7:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.8)))
    jpeg_q = rng.randint(45, 92)
    return img, new_geoms, jpeg_q, fwd


# ---------------------------------------------------------------------------
# Stage 8 — OUTPUT (PAGE XML, same dialect as generate_page_xml.py)
# ---------------------------------------------------------------------------
def pts(seq) -> str:
    return " ".join(f"{int(round(x))},{int(round(y))}" for x, y in seq)


def write_page_xml(out_path: Path, image_name: str, W: int, H: int,
                   geoms: list[LineGeom], fwd, img_dirname: str = "images") -> None:
    pcgts = ET.Element(f"{{{PAGE_NS}}}PcGts")
    md = ET.SubElement(pcgts, f"{{{PAGE_NS}}}Metadata")
    ET.SubElement(md, f"{{{PAGE_NS}}}Creator").text = "OCRInscriptiones/generate_synthetic.py"
    ET.SubElement(md, f"{{{PAGE_NS}}}Created").text = ""
    ET.SubElement(md, f"{{{PAGE_NS}}}LastChange").text = ""
    page = ET.SubElement(pcgts, f"{{{PAGE_NS}}}Page",
                         imageFilename=f"../{img_dirname}/{image_name}",
                         imageWidth=str(W), imageHeight=str(H))
    region = ET.SubElement(page, f"{{{PAGE_NS}}}TextRegion", id="r1", type="paragraph")
    ET.SubElement(region, f"{{{PAGE_NS}}}Coords",
                  points=pts([(0, 0), (W, 0), (W, H), (0, H)]))
    for i, g in enumerate(geoms, 1):
        line_el = ET.SubElement(region, f"{{{PAGE_NS}}}TextLine", id=f"r1l{i}")
        # mask polygon: transformed baseline offset vertically by the line's
        # untransformed ascender/descender (exact under small-angle affine)
        asc = g.baseline[0][1] - g.top
        desc = g.bottom - g.baseline[0][1]
        upper = [(x, y - asc) for x, y in g.baseline]
        lower = [(x, y + desc) for x, y in reversed(g.baseline)]
        ET.SubElement(line_el, f"{{{PAGE_NS}}}Coords", points=pts(upper + lower))
        ET.SubElement(line_el, f"{{{PAGE_NS}}}Baseline", points=pts(g.baseline))
        te = ET.SubElement(line_el, f"{{{PAGE_NS}}}TextEquiv")
        ET.SubElement(te, f"{{{PAGE_NS}}}Unicode").text = g.gt
    tree = ET.ElementTree(pcgts)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def write_review_html(out: Path, entries: list[dict], limit: int = 80) -> None:
    rows = []
    for e in entries[:limit]:
        gt_html = "<br>".join(html.escape(l) for l in e["lines"])
        rows.append(
            f'<div class="card"><img src="images/{e["img"]}" loading="lazy">'
            f'<div class="meta"><b>{e["img"]}</b> · {html.escape(e["hd_nr"])} · '
            f'{html.escape(e["province"])} · {e["stone"]} · {e["font"]}</div>'
            f'<pre>{gt_html}</pre></div>')
    out.write_text(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Synthetic corpus review</title><style>
 body {{ background:#1a1a22; color:#ddd; font-family:sans-serif; }}
 .card {{ display:inline-block; vertical-align:top; width:430px;
          margin:8px; background:#26262e; border-radius:6px; padding:8px; }}
 .card img {{ width:100%; border-radius:4px; }}
 .meta {{ font-size:11px; color:#9a9; margin:4px 0; }}
 pre {{ font-size:12px; color:#cdc; white-space:pre-wrap; }}
</style></head><body>
<h2>Synthetic corpus — first {min(limit, len(entries))} of {len(entries)}</h2>
{''.join(rows)}</body></html>""", encoding="utf-8")


# ---------------------------------------------------------------------------
# ------------------------------------------------------------------ parallel
# Module-level worker plumbing so multiprocessing works on both fork (Linux/WSL)
# and spawn (Windows). Read-only state is set per worker via the initializer.
_CORPUS = None
_WIDTH = 900
_SEED = 42
_IMG_DIR = None
_PAGE_DIR = None
_STONE_KINDS = None
_STONE_W = None
_DOTS = False


def _init_worker(corpus, width, seed, img_dir, page_dir, dots=False):
    global _CORPUS, _WIDTH, _SEED, _IMG_DIR, _PAGE_DIR, _STONE_KINDS, _STONE_W, _DOTS
    _CORPUS, _WIDTH, _SEED = corpus, width, seed
    _IMG_DIR, _PAGE_DIR = Path(img_dir), Path(page_dir)
    _STONE_KINDS, _STONE_W = zip(*[(k, v["w"]) for k, v in STONE_PALETTES.items()])
    _DOTS = dots


def render_one(job):
    """Render a single synthetic inscription. job = (out_index, corpus_index).
    Deterministic per out_index (seed = base_seed*P + out_index), resume-safe."""
    gidx, ci = job
    name = f"synth_{gidx:05d}"
    out_img = _IMG_DIR / f"{name}.jpg"
    out_xml = _PAGE_DIR / f"{name}.xml"
    if out_img.exists() and out_xml.exists():
        return None  # already generated — skip (resume / parallel-safe)
    insc = _CORPUS[ci]
    seed = (_SEED * 1_000_003 + gidx) & 0x7FFFFFFF
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)

    W = _WIDTH
    n_lines = len(insc["lines"])
    H = int(W * rng.uniform(0.22 + 0.06 * n_lines, 0.32 + 0.085 * n_lines))
    H = max(220, min(H, int(W * 1.25)))

    font_path = pick_font(rng)
    # dots mode: scriptio continua with middot · between words (rendered glyph + GT);
    # else keep spaces (rendered as gaps; stripped to letter-only at training).
    lines = ["·".join(l.split()) for l in insc["lines"]] if _DOTS else insc["lines"]
    frag = make_fragment_mask(W, H, rng, nrng) if rng.random() < 0.35 else None
    mask, geoms, dmg = render_text_mask(lines, W, H, rng, font_path, frag)
    if not geoms and frag is not None:
        frag = None
        mask, geoms, dmg = render_text_mask(lines, W, H, rng, font_path)
    if not geoms:
        return None
    if rng.random() < 0.55:
        stone = load_real_background(W, H, rng)
        kind = "real-bg"
        if stone is None:
            kind = rng.choices(_STONE_KINDS, weights=_STONE_W, k=1)[0]
            stone = make_stone(W, H, kind, nrng)
    else:
        kind = rng.choices(_STONE_KINDS, weights=_STONE_W, k=1)[0]
        stone = make_stone(W, H, kind, nrng)
    mean_ascent = float(np.mean([g.baseline[0][1] - g.top for g in geoms]))
    carved = carve(stone, mask, nrng, damage=dmg, font_px=mean_ascent / 0.9)
    if frag is not None:
        carved = apply_fracture(carved, frag, nrng)
    img, geoms2, jpeg_q, fwd = camera(carved, geoms, rng, nrng)

    img.save(out_img, "JPEG", quality=jpeg_q)
    write_page_xml(out_xml, f"{name}.jpg", W, H, geoms2, fwd)
    return {"img": f"{name}.jpg", "hd_nr": insc["hd_nr"], "province": insc["province"],
            "stone": kind + (" +frag" if frag is not None else ""),
            "font": font_path.stem, "lines": [g.gt for g in geoms2]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus", default=str(REPO / "edh_inscriptions.json"))
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--start", type=int, default=0,
                    help="output index offset (e.g. 6135 to append after existing synth)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes (CPU-bound generation)")
    ap.add_argument("--outdir", default=str(ROOT),
                    help="output dir holding images/ and page/ (default: synthetic/)")
    ap.add_argument("--dots", action="store_true",
                    help="render middot · between words (scriptio continua); else spaces")
    ap.add_argument("--keep-photos", action="store_true",
                    help="do NOT drop photographed inscriptions (default: keep only no-photo)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)

    img_dir = Path(args.outdir) / "images"
    page_dir = Path(args.outdir) / "page"
    img_dir.mkdir(parents=True, exist_ok=True)
    page_dir.mkdir(parents=True, exist_ok=True)

    # exclude our real annotated HD numbers (belt-and-suspenders vs the photo filter)
    exclude = set()
    for lf in ("cluster/bundle/lists/recog_train_real.txt", "cluster/bundle/lists/recog_val.txt"):
        p = REPO / lf
        if p.exists():
            for s in p.read_text().split():
                if s.strip():
                    exclude.add(s.strip().split("/")[-1].split("_")[0].replace(".xml", ""))
    corpus = load_corpus(Path(args.corpus), rng,
                         no_photos_only=not args.keep_photos,
                         exclude_hds=frozenset(exclude))
    if len(corpus) < args.n:
        print(f"WARNING: only {len(corpus)} usable inscriptions for n={args.n}")

    # capped base selection (<=20% supplied-bracket), then cycle to args.n so we
    # can request more than the corpus size (texts repeat, rendering differs).
    max_br = int(len(corpus) * 0.20)
    br_used = 0
    base_idx: list[int] = []
    for i, insc in enumerate(corpus):
        if insc.get("has_br"):
            if br_used >= max_br:
                continue
            br_used += 1
        base_idx.append(i)
    jobs = [(args.start + j, base_idx[j % len(base_idx)]) for j in range(args.n)]
    print(f"  corpus={len(corpus)} base={len(base_idx)} (br cap {max_br}) -> "
          f"{len(jobs)} jobs as synth_{args.start:05d}..synth_{args.start + args.n - 1:05d}  "
          f"workers={args.workers}")

    entries = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(corpus, args.width, args.seed,
                               str(img_dir), str(page_dir), args.dots)) as pool:
            for k, e in enumerate(pool.imap_unordered(render_one, jobs, chunksize=8), 1):
                if e:
                    entries.append(e)
                if k % 200 == 0:
                    print(f"  [{k}/{len(jobs)}]")
    else:
        _init_worker(corpus, args.width, args.seed, str(img_dir), str(page_dir), args.dots)
        for k, job in enumerate(jobs, 1):
            e = render_one(job)
            if e:
                entries.append(e)
            if k % 200 == 0:
                print(f"  [{k}/{len(jobs)}]")

    # rebuild the COMPLETE train list from everything on disk
    all_xml = sorted(page_dir.glob("synth_*.xml"))
    (ROOT / "synth_xml.txt").write_text(
        "\n".join(f"synthetic/page/{p.name}" for p in all_xml) + "\n", encoding="utf-8")
    write_review_html(ROOT / "review.html", entries[:300])
    print(f"\nDone. generated/updated {len(entries)} this run; "
          f"total synth on disk: {len(all_xml)}")
    print(f"Train list: {ROOT / 'synth_xml.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
