r"""
NLB maize (Prashanth et al., WACV 2024) -> lesion masks for Paper 1. Fully automatic, no manual work.

The authors sent one mask PER LESION (4682 masks for 999 images). This script joins all lesion masks of an
image into one binary mask (lesion = 255, rest = 0) and writes the images/ + annotations/ layout that
p1lib.py already reads (the same layout as PlantSeg). The 70/15/15 train/val/test split is made later by
p1lib itself (leak-free, near-duplicate photos kept together), exactly as for PlantSeg.

  conda activate cropai
  python nlb_wacv_prepare.py check      # 1) look first: file names, how masks match images, sizes
  python nlb_wacv_prepare.py build      # 2) write the merged dataset + preview.jpg + summary.txt

Input :  <P1_ROOT>\Dataset\nlb_wacv\all_images_wacv\      (999 photos)
         <P1_ROOT>\Dataset\nlb_wacv\disease_masks_WACV\   (4682 lesion masks)
Output:  <P1_ROOT>\Dataset\nlbseg_wacv\images\*.jpg
         <P1_ROOT>\Dataset\nlbseg_wacv\annotations\*.png
         <P1_ROOT>\Dataset\nlbseg_wacv\summary.txt, preview.jpg, match.csv
"""
import collections, random, re, sys
import os
from pathlib import Path
import numpy as np, pandas as pd, cv2

SRC = Path(os.environ.get("P1_ROOT", ".")) / r"Dataset/nlb_wacv"
IMG_DIR, MSK_DIR = SRC / "all_images_wacv", SRC / "disease_masks_WACV"
OUT = Path(os.environ.get("P1_ROOT", ".")) / r"Dataset/nlbseg_wacv"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def files(d):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in EXT)


def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower())          # case and separator style ignored, separators kept


def match():
    """Each mask -> the image whose name is the longest start of the mask's name (e.g. IMG_12.jpg <- IMG_12_3.png).
    Separators and letter case are ignored. Exact name matches win."""
    imgs, msks = files(IMG_DIR), files(MSK_DIR)
    by_norm = {}
    for p in imgs:
        by_norm.setdefault(norm(p.stem), p)
    keys = sorted(by_norm, key=len, reverse=True)
    out, unmatched = collections.defaultdict(list), []
    for m in msks:
        k = norm(m.stem)
        hit = by_norm.get(k) or by_norm.get(re.sub(r"^disease_mask_\d+_", "", k))   # WACV naming: disease_mask_<n>_<image>
        if hit is None:
            # longest image name that the mask name starts with, and the rest must not continue a number
            # (so IMG_1 does not swallow IMG_12_0)
            for key in keys:
                nxt = k[len(key):len(key) + 1]
                same_kind = (key[-1:].isdigit() and nxt.isdigit()) or (key[-1:].isalpha() and nxt.isalpha())
                if k.startswith(key) and not same_kind:            # IMG_1 matches IMG_1_0, not IMG_12_0
                    hit = by_norm[key]; break
        if hit is None and m.parent != MSK_DIR:              # masks stored in one folder per image
            hit = by_norm.get(norm(m.parent.name))
        (out[hit].append(m) if hit is not None else unmatched.append(m))
    return imgs, msks, out, unmatched


def read_pair(img_p, mask_ps):
    """Image in the orientation of its masks (some cameras store a rotation flag), and the merged lesion mask."""
    ms = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in mask_ps]
    ms = [m for m in ms if m is not None]
    if not ms:
        return None, None, "masks unreadable"
    ms = [(m.reshape(m.shape[0], m.shape[1], -1)[..., :3].max(-1) > 0) if m.ndim == 3 else (m > 0) for m in ms]
    shp = collections.Counter(m.shape for m in ms).most_common(1)[0][0]
    img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
    note = ""
    if img is not None and img.shape[:2] != shp:
        raw = cv2.imread(str(img_p), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if raw is not None and raw.shape[:2] == shp:
            img, note = raw, "used raw orientation"
    if img is None:
        return None, None, "image unreadable"
    H, W = img.shape[:2]
    merged = np.zeros((H, W), np.uint8)
    for m in ms:
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            note = note or "mask resized"
        merged |= m.astype(np.uint8)
    return img, merged * 255, note


def check():
    imgs, msks, out, unmatched = match()
    print(f"images: {len(imgs)} | lesion masks: {len(msks)}")
    print("first image names:", [p.name for p in imgs[:5]])
    print("first mask names :", [p.relative_to(MSK_DIR).as_posix() for p in msks[:8]])
    n = [len(v) for v in out.values()]
    print(f"masks matched to an image: {sum(n)} / {len(msks)} | images with >=1 mask: {len(out)} / {len(imgs)}")
    if n:
        print(f"lesion masks per image: min {min(n)}, median {int(np.median(n))}, max {max(n)}")
    if unmatched:
        print(f"UNMATCHED masks: {len(unmatched)}, e.g. {[p.name for p in unmatched[:5]]}  <- send me this line")
    no_mask = [p.name for p in imgs if p not in out]
    if no_mask:
        print(f"images without any mask: {len(no_mask)}, e.g. {no_mask[:5]}")
    for img_p in random.Random(0).sample(sorted(out), min(3, len(out))):
        im, m, note = read_pair(img_p, out[img_p])
        print(f"  {img_p.name}: {len(out[img_p])} masks | image {None if im is None else im.shape[:2]} | lesion share "
              f"{0 if m is None else (m > 0).mean():.2%} {note}")
    ok = not unmatched and len(out) >= 0.95 * len(imgs)
    print("\nLOOKS OK -> run: python nlb_wacv_prepare.py build" if ok else "\nNOT OK -> send me this whole output")


def build():
    imgs, msks, out, unmatched = match()
    (OUT / "images").mkdir(parents=True, exist_ok=True); (OUT / "annotations").mkdir(parents=True, exist_ok=True)
    rows, notes = [], collections.Counter()
    for i, img_p in enumerate(sorted(out), 1):
        im, m, note = read_pair(img_p, out[img_p])
        if im is None:
            notes[note] += 1; continue
        if note:
            notes[note] += 1
        key = re.sub(r"[^A-Za-z0-9_-]", "_", img_p.stem)
        cv2.imwrite(str(OUT / "images" / f"{key}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(OUT / "annotations" / f"{key}.png"), m)
        n_les = cv2.connectedComponents((m > 0).astype(np.uint8))[0] - 1
        rows.append(dict(key=key, source_image=img_p.name, n_mask_files=len(out[img_p]), n_lesion_regions=n_les,
                         lesion_frac=float((m > 0).mean()), height=im.shape[0], width=im.shape[1]))
        if i % 100 == 0:
            print(f"  {i}/{len(out)} images")
    df = pd.DataFrame(rows); df.to_csv(OUT / "match.csv", index=False)
    s = (f"images written {len(df)} (of {len(imgs)}; {len(imgs) - len(out)} had no mask)\n"
         f"lesion mask files used {int(df.n_mask_files.sum())} of {len(msks)} (unmatched {len(unmatched)})\n"
         f"lesion share per image: mean {df.lesion_frac.mean():.2%}, median {df.lesion_frac.median():.2%}\n"
         f"lesion regions per image: mean {df.n_lesion_regions.mean():.1f}\n"
         f"image sizes: {df.groupby(['width', 'height']).size().sort_values(ascending=False).head(3).to_dict()}\n"
         f"notes: {dict(notes)}\n")
    (OUT / "summary.txt").write_text(s); print(s)
    vis = []
    for _, r in df.sample(min(8, len(df)), random_state=1).iterrows():
        im = cv2.imread(str(OUT / "images" / f"{r.key}.jpg")); m = cv2.imread(str(OUT / "annotations" / f"{r.key}.png"), 0)
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnt, -1, (0, 255, 255), max(2, im.shape[1] // 300))
        vis.append(cv2.resize(im, (450, 300)))
    if vis:
        while len(vis) % 4:
            vis.append(np.zeros_like(vis[0]))
        grid = np.vstack([np.hstack(vis[k:k + 4]) for k in range(0, len(vis), 4)])
        cv2.imwrite(str(OUT / "preview.jpg"), grid); print("preview (yellow = lesion outline):", OUT / "preview.jpg")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"check": check, "build": build}.get(cmd, lambda: print(__doc__))()
