r"""
Qualitative figure for the Paper 1 manuscript (Fig. 4). Reads the trained seed-0 models from both run folders,
picks test images AT RANDOM (fixed seed, fixed rule: lesion share 3-40% of the image, so lesions are visible at
print size; no cherry-picking), and draws image / ground truth / student alone / baseline / ours / teacher
for clean and heavy-clutter versions.

  conda activate cropai
  cd <repo>/scripts
  python qual_figure.py            # ~2 min on the GPU (works on CPU too, slower)

Output: <P1_ROOT>\paper1_figures\fig_qualitative.pdf and .png
        <P1_ROOT>\paper1_figures\fig_qualitative_picks.csv    (which images were used)
Colours: yellow = correct lesion pixels, red = false lesion (false positive), cyan = missed lesion (false negative).
"""
import sys
import os
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "p1lib"))
import p1lib as L

OUT = Path(os.environ.get("P1_ROOT", ".")) / r"paper1_figures"
SETS = [  # run folder, label, baseline method, rows: (n images, test set)
    (Path(os.environ.get("P1_ROOT", ".")) / r"paper1_runs_v2", "PlantSeg", "B_Caug", [(2, "clutter_heavy"), (1, "clean")]),
    (Path(os.environ.get("P1_ROOT", ".")) / r"paper1_runs_nlb", "NLB", "B_CWD", [(1, "clutter_heavy")]),
]
STU = "mobilenetv3_small_100"
SEED = 0


def overlay(img, pred, gt):
    base = (img * 0.55).astype(np.uint8).copy()
    tp, fp, fn = (pred == 1) & (gt == 1), (pred == 1) & (gt == 0), (pred == 0) & (gt == 1)
    for m, c in ((tp, (255, 214, 0)), (fp, (230, 40, 40)), (fn, (0, 220, 255))):
        base[m] = (0.35 * base[m] + 0.65 * np.array(c)).astype(np.uint8)
    return base


def iou(p, g):
    u = ((p == 1) | (g == 1)).sum()
    return 100.0 * ((p == 1) & (g == 1)).sum() / u if u else 100.0


@torch.no_grad()
def predict(model, img, dev):
    return model(L.to_tensor(img)[None].to(dev)).argmax(1)[0].cpu().numpy().astype(np.uint8)


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows, picks = [], []
    for work, label, base, plan in SETS:
        cfg = L.load_cfg(work); cfg["paths"]["work"] = str(work); s = cfg["img_size"]
        test = L.load_split_frames(cfg)["test"]
        cand = test[(test.lesion_frac >= 0.03) & (test.lesion_frac <= 0.40)].sample(frac=1.0, random_state=SEED)
        heavy = pd.read_csv(work / "clutter_test" / "heavy" / "index.csv", dtype={"key": str}).set_index("key")
        runs = {"Student alone": f"{STU}__A0_student__s0", ("C-aug" if base == "B_Caug" else "CWD"): f"{STU}__{base}__s0",
                "Ours (LSD+CPD)": f"{STU}__A7_full__s0", "Teacher": L.teacher_run(cfg, cfg["teacher_choice"]) if hasattr(L, "teacher_run") else None}
        models = {k: L.load_model(cfg, v, dev)[0] for k, v in runs.items() if v}
        k = 0
        for n, kind in plan:
            for _ in range(n):
                r = cand.iloc[k]; k += 1
                if kind == "clean":
                    img, gt = L.read_pair(r)
                    img = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA); gt = cv2.resize(gt, (s, s), interpolation=cv2.INTER_NEAREST)
                else:
                    h = heavy.loc[str(r.key)]
                    img = cv2.cvtColor(cv2.imread(h.image), cv2.COLOR_BGR2RGB); gt = (cv2.imread(h.mask, 0) > 0).astype(np.uint8)
                preds = {name: predict(m, img, dev) for name, m in models.items()}
                rows.append((f"{label}, {'heavy clutter' if kind != 'clean' else 'clean'}", img, gt, preds))
                picks.append(dict(dataset=label, test=kind, key=r.key, lesion_frac=r.lesion_frac))
    cols = ["Image", "Ground truth"] + list(rows[0][3].keys())
    fig, ax = plt.subplots(len(rows), len(cols), figsize=(1.2 * len(cols), 1.25 * len(rows)))
    for i, (title, img, gt, preds) in enumerate(rows):
        panels = [(img, ""), (overlay(img, gt, gt), "")] + [(overlay(img, p, gt), f"IoU {iou(p, gt):.0f}") for p in preds.values()]
        for j, (im, txt) in enumerate(panels):
            a = ax[i, j]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
            if txt:
                a.text(0.03, 0.04, txt, transform=a.transAxes, fontsize=6, color="white", bbox=dict(fc="black", alpha=0.6, pad=1, lw=0))
            if i == 0:
                a.set_title(cols[j], fontsize=7)
            if j == 0:
                a.set_ylabel(title, fontsize=6.5)
    fig.tight_layout(pad=0.2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_qualitative.pdf", bbox_inches="tight"); fig.savefig(OUT / "fig_qualitative.png", dpi=250, bbox_inches="tight")
    pd.DataFrame(picks).to_csv(OUT / "fig_qualitative_picks.csv", index=False)
    print("saved", OUT / "fig_qualitative.pdf")


if __name__ == "__main__":
    main()
