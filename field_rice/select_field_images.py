"""Pick the Paper 1 field-test images: leaf classes only, one photo per burst group, GPS and all metadata removed."""
import sys, os, random, csv
import numpy as np
from PIL import Image, ImageOps

src, out = sys.argv[1], sys.argv[2]
PER_CLASS = 15
LEAF = ["Bacterial_leaf_blight", "Brown_Spot_Leaf", "Rice_Blast", "Sheath_blight"]
os.makedirs(f"{out}/images", exist_ok=True)
rows = []
for c in LEAF:
    fs = sorted(f for f in os.listdir(f"{src}/{c}") if f.lower().endswith(".jpg"))
    H = []
    for f in fs:
        im = Image.open(f"{src}/{c}/{f}"); im.draft("L", (256, 256))
        a = np.asarray(im.convert("L").resize((9, 8)), dtype=np.int16); H.append((a[:, 1:] > a[:, :-1]).flatten())
    H = np.array(H); D = (H[:, None] != H[None]).sum(-1); n = len(fs); par = list(range(n))
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= 8 or (j == i + 1 and D[i, j] <= 12):   # near-duplicates and consecutive burst shots
                par[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    rng = random.Random(0)
    reps = [rng.choice(g) for g in groups.values()]
    pick = sorted(rng.sample(reps, min(PER_CLASS, len(reps))))
    print(f"{c}: {n} photos, {len(groups)} distinct scenes, {len(pick)} selected")
    for k, i in enumerate(pick):
        name = f"{c}_{k + 1:02d}.jpg"
        im = ImageOps.exif_transpose(Image.open(f"{src}/{c}/{fs[i]}")).convert("RGB")
        im.thumbnail((2048, 2048), Image.LANCZOS)
        im.save(f"{out}/images/{name}", quality=95)          # saved without EXIF: no GPS, no camera data
        rows.append(dict(file=name, disease=c, original=fs[i]))
with open(f"{out}/field_images.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["file", "disease", "original"]); w.writeheader(); w.writerows(rows)
print("total", len(rows))
