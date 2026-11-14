r"""
REV-3 and REV-4 for Paper 1 (reviewer requests). Evaluation only: no training, nothing is changed in the runs.

REV-3  Held-out robustness test. The clutter test in the paper uses the same occluder bank as training. Here we
       build four NEW test versions of the 1,720 PlantSeg test images that training never saw:
         occ_val     heavy clutter with occluders/backgrounds cut from VALIDATION images (not the training bank)
         motion_blur directional motion blur (kernel 9-21 px, random angle)  - corruption type not used in training
         noise       Gaussian sensor noise (sigma 15-25 grey levels)         - not used in training
         haze        fog / haze (blend toward light grey, strength 0.35-0.55) - not used in training
REV-4  Severity check on the clean test set: leaf area = leaf proxy UNION lesion pixels (so lesions are never
       outside the leaf), severity = lesion / leaf area, and agreement in severity classes 0-5 / 5-25 / >25 %
       (accuracy, quadratic-weighted kappa), plus bias, Bland-Altman limits and Lin's concordance (CCC).

Uses p1lib/p1lib_rev.py (found automatically). Uses the GPU, ~1-2 h in total. Do not train at the same time.

  conda activate cropai
  cd <repo>/scripts
  python rev34_eval.py build      # 1) builds the 4 held-out test sets (~10 min, CPU)
  python rev34_eval.py eval       # 2) runs every trained model on them + clean test (~1-2 h, GPU); resumable
  python rev34_eval.py summary    # 3) tables -> <work>\results\table_rev3_heldout.csv, table_rev4_severity.csv
                                  #    (+ table_rev3_per_seed.csv)
"""
import sys, json
import os
from pathlib import Path
import numpy as np, pandas as pd, cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "p1lib"))
import p1lib_rev as L

WORK = Path(os.environ.get("P1_ROOT", ".")) / r"paper1_runs_v2"
SETS = ["occ_val", "motion_blur", "noise", "haze"]
METHODS = ["A0_student", "B_CWD", "B_BPKD", "B_SARD", "B_Caug", "R4_augOnly", "R6_augCWD", "A6_CPD", "A7_full"]
SEEDS = [0, 1, 2]
CLASSES = [0, 5, 25, 101]                      # severity classes in % of leaf area: [0,5), [5,25), [25,100]


def setup(work):
    cfg = L.load_cfg(work); cfg["paths"]["work"] = str(work)
    return cfg


def runs(cfg):
    names = []
    for st in cfg["students"]:
        for m in METHODS:
            for sd in SEEDS:
                n = f"{L.short(st)}__{m}__s{sd}"
                if L.run_done(cfg, n):
                    names.append(n)
    t = L.chosen_teacher(cfg) if hasattr(L, "chosen_teacher") else None
    t = t if (t and str(t).startswith("teacher-")) else (L.teacher_run(cfg, t) if t else None)
    if t and L.run_done(cfg, t):
        names.append(t)
    return names


# ----------------------------------------------------------------------------------------- REV-3 test sets
def motion_blur(img, rng):
    k = int(rng.integers(9, 22)); ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
    ker = cv2.warpAffine(ker, rot, (k, k)); ker /= max(ker.sum(), 1e-6)
    return cv2.filter2D(img, -1, ker)


def noise(img, rng):
    return np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(15, 25), img.shape), 0, 255).astype(np.uint8)


def haze(img, rng):
    a = rng.uniform(0.35, 0.55)
    return np.clip(img.astype(np.float32) * (1 - a) + a * 215, 0, 255).astype(np.uint8)


def build(work):
    from tqdm.auto import tqdm
    cfg = setup(work); s = cfg["img_size"]; frames = L.load_split_frames(cfg); test = frames["test"]
    val_bank = L.make_bank(frames["val"], cfg, n=300, seed=1)          # validation images only: unseen by training
    for name, sd in [("occ_val", 44), ("motion_blur", 55), ("noise", 66), ("haze", 77)]:
        d = L.wp(cfg, "heldout_test", name); csv = d / "index.csv"
        if csv.exists():
            print("exists:", d); continue
        (d / "images").mkdir(parents=True, exist_ok=True); (d / "masks").mkdir(exist_ok=True)
        rng = np.random.default_rng(sd); cl = L.Clutter(val_bank, p=1.0, level="heavy", seed=sd); rows = []
        for _, r in tqdm(test.iterrows(), total=len(test), desc=name):
            img, msk = L.read_pair(r)
            img = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA); msk = cv2.resize(msk, (s, s), interpolation=cv2.INTER_NEAREST)
            if name == "occ_val":
                img, msk, _ = cl(img, msk)
            else:
                img = {"motion_blur": motion_blur, "noise": noise, "haze": haze}[name](img, rng)
            ip, mp = d / "images" / f"{r.key}.png", d / "masks" / f"{r.key}.png"
            cv2.imwrite(str(ip), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)); cv2.imwrite(str(mp), msk * 255)
            rows.append(dict(image=str(ip), mask=str(mp), key=r.key, disease=r.disease, lesion_frac=r.lesion_frac))
        pd.DataFrame(rows).to_csv(csv, index=False); print("built", d)
    print("next: python rev34_eval.py eval")


# ----------------------------------------------------------------------------------------- evaluation
def load_set(cfg, name, frames):
    s = cfg["img_size"]
    fr = frames["test"] if name == "clean" else pd.read_csv(L.wp(cfg, "heldout_test", name, "index.csv"), dtype={"key": str})
    imgs, ys, keys, leaves = [], [], [], []
    for _, r in fr.iterrows():
        img, y = L.read_pair(r)
        if img.shape[0] != s or img.shape[1] != s:
            img = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA); y = cv2.resize(y, (s, s), interpolation=cv2.INTER_NEAREST)
        imgs.append(img); ys.append(y); keys.append(str(r.key)); leaves.append(L.leaf_proxy(img) > 0)
    return imgs, ys, keys, leaves


def evaluate(work):
    import torch
    cfg = setup(work); frames = L.load_split_frames(cfg)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out = L.wp(cfg, "results", "per_image_rev34"); out.mkdir(parents=True, exist_ok=True)
    names = runs(cfg); print(f"{len(names)} trained models found:", ", ".join(sorted({n.split('__')[1] for n in names})))
    for sname in ["clean"] + SETS:
        todo = [n for n in names if not (out / f"{n}__{sname}.csv").exists()]
        if not todo:
            print("done:", sname); continue
        imgs, ys, keys, leaves = load_set(cfg, sname, frames); print(f"{sname}: {len(imgs)} images, {len(todo)} models to run")
        for n in todo:
            m, _ = L.load_model(cfg, n, dev); rows = []
            with torch.no_grad():
                for i in range(0, len(imgs), 16):
                    x = torch.stack([L.to_tensor(im) for im in imgs[i:i + 16]]).to(dev)
                    pred = m(x).argmax(1).cpu().numpy().astype(np.uint8)
                    for k, p in enumerate(pred):
                        y, lf = ys[i + k], leaves[i + k]
                        rows.append(dict(key=keys[i + k], inter1=int(((p == 1) & (y == 1)).sum()), union1=int(((p == 1) | (y == 1)).sum()),
                                         inter0=int(((p == 0) & (y == 0)).sum()), union0=int(((p == 0) | (y == 0)).sum()),
                                         gt_px=int(y.sum()), pred_px=int(p.sum()), leaf_px=int(lf.sum()),
                                         leaf_gt_px=int((lf | (y == 1)).sum()), leaf_pred_px=int((lf | (p == 1)).sum())))
            d = pd.DataFrame(rows); d["iou"] = np.where(d.union1 > 0, d.inter1 / d.union1.clip(lower=1), 1.0)
            L.atomic_to_csv(d, out / f"{n}__{sname}.csv", index=False)
            del m
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  {n} {sname}: mIoU {miou(d):.2f}")
    print("next: python rev34_eval.py summary")


def miou(d):
    return 100 * (d.inter1.sum() / max(d.union1.sum(), 1) + d.inter0.sum() / max(d.union0.sum(), 1)) / 2


def ccc(a, b):
    ma, mb, va, vb = a.mean(), b.mean(), a.var(), b.var()
    return 2 * ((a - ma) * (b - mb)).mean() / (va + vb + (ma - mb) ** 2)


def wkappa(t, p, k=3):
    O = np.zeros((k, k)); [O.__setitem__((i, j), O[i, j] + 1) for i, j in zip(t, p)]
    W = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / max((W * E).sum(), 1e-9)


def summary(work):
    cfg = setup(work); out = L.wp(cfg, "results", "per_image_rev34"); res = L.wp(cfg, "results")
    files = sorted(out.glob("*.csv"))
    if not files:
        sys.exit("no results yet: run  python rev34_eval.py eval")
    rows, sev = [], []
    for f in files:
        run, sname = f.stem.rsplit("__", 1); model, method, seed = run.split("__")
        d = pd.read_csv(f, dtype={"key": str})
        rows.append(dict(model=model, method=method, seed=int(seed[1:]), test=sname, miou=miou(d)))
        if sname == "clean":
            t = 100 * d.gt_px / d.leaf_gt_px.clip(lower=1); p = 100 * d.pred_px / d.leaf_pred_px.clip(lower=1)
            tc, pc = np.digitize(t, CLASSES[1:-1]), np.digitize(p, CLASSES[1:-1]); diff = p - t
            sev.append(dict(model=model, method=method, seed=int(seed[1:]), mae=diff.abs().mean(), bias=diff.mean(),
                            loa_low=diff.mean() - 1.96 * diff.std(), loa_high=diff.mean() + 1.96 * diff.std(), ccc=ccc(t, p),
                            class_acc=100 * (tc == pc).mean(), wkappa=wkappa(tc, pc), n=len(d),
                            share_class=" / ".join(f"{100 * (tc == k).mean():.0f}" for k in range(3))))
    R = pd.DataFrame(rows); S = pd.DataFrame(sev)
    piv = R.pivot_table(index=["model", "method", "seed"], columns="test", values="miou").reset_index()
    piv.to_csv(res / "table_rev3_per_seed.csv", index=False)
    cols = [c for c in ["clean"] + SETS if c in piv]
    T = piv.groupby(["model", "method"])[cols].agg(["mean", "std"])
    T.columns = [f"{a}_{b}" for a, b in T.columns]; T = T.round(2).reset_index()
    T.to_csv(res / "table_rev3_heldout.csv", index=False)
    print("\nREV-3 held-out robustness (mIoU %, mean / s.d. over seeds)\n", T.to_string(index=False))
    if len(S):
        num = ["mae", "bias", "loa_low", "loa_high", "ccc", "class_acc", "wkappa"]
        G = S.groupby(["model", "method"])[num].agg(["mean", "std"]); G.columns = [f"{a}_{b}" for a, b in G.columns]
        G = G.round(3).reset_index(); G["true_class_share_%"] = S.groupby(["model", "method"]).share_class.first().values
        G.to_csv(res / "table_rev4_severity.csv", index=False)
        print("\nREV-4 severity on clean test (leaf = proxy U lesion; classes 0-5/5-25/>25 %)\n", G.to_string(index=False))
    print("\nsaved", res / "table_rev3_heldout.csv", "+ table_rev3_per_seed.csv + table_rev4_severity.csv")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    work = Path(sys.argv[2]) if len(sys.argv) > 2 else WORK
    {"build": build, "eval": evaluate, "summary": summary}.get(cmd, lambda w: print(__doc__))(work)
