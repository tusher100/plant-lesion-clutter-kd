"""
finetune_field.py  -  Paper 1: few-shot adaptation of the on-device students to rice field photos.

3-fold cross-validation over the 60 labelled photos (stratified by disease: 40 train / 20 test per fold,
every photo is tested exactly once). For each starting point (PlantSeg checkpoints A0 / R4 / R6 / A7,
Lite0 A0 / A7, the teacher, and ImageNet-only), we
  1. score it zero-shot on the test fold ("before"),
  2. fine-tune it on the 40 training photos for a FIXED schedule (no early stopping, test fold never seen),
  3. score it again ("after").
Metrics are the same as field_eval.py (pooled lesion IoU, in-leaf lesion IoU, severity MAE / kappa).

Run from <repo> (needs p1lib_rev.py / p1lib.py and field_eval.py there):
    python finetune_field.py --work "<P1_ROOT>\\paper1_runs_v2" --field "<P1_ROOT>\\field_test"
Options: --quick (1 seed, mbv3 only, to test), --epochs 60, --size 320, --no-teacher
Resumable: finished (init, seed, fold) runs are cached in FIELD/results_ft/runs/.
Outputs in FIELD/results_ft/: ft_summary.csv (main table), ft_by_disease.csv, ft_paired.csv, ft_table.tex
"""
import argparse, glob, os, sys, time
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.getcwd())
import field_eval as fe  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


# ------------------------------------------------------------------ data
def load_items(field, keep_res=1024):
    meta = pd.read_csv(os.path.join(field, "field_images.csv"))
    dis = dict(zip(meta.file, meta.disease))
    items = []
    for p in sorted(glob.glob(os.path.join(field, "images", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        im = Image.open(p).convert("RGB")
        les, leaf = fe.load_labels(os.path.join(field, "labels"), stem, im.size)
        if les is None or leaf is None:
            continue
        s = keep_res / max(im.size)
        sz = (round(im.width * s), round(im.height * s))
        items.append(dict(file=os.path.basename(p), disease=dis.get(os.path.basename(p), stem.rsplit("_", 1)[0]),
                          img=np.array(im.resize(sz, Image.BILINEAR)),
                          les=np.array(Image.fromarray(les * 255).resize(sz, Image.NEAREST)) > 0,
                          leaf=np.array(Image.fromarray(leaf * 255).resize(sz, Image.NEAREST)) > 0))
    return items


def folds_by_disease(items, k=3, seed=0):
    r = np.random.default_rng(seed); fold = {}
    for d in sorted({it["disease"] for it in items}):
        idx = [i for i, it in enumerate(items) if it["disease"] == d]
        r.shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = j % k
    return [fold[i] for i in range(len(items))]


def to_eval(it, size):
    rs = lambda a, m: np.array(Image.fromarray(a).resize((size, size), m))  # noqa: E731
    rgb = rs(it["img"], Image.BILINEAR)
    les = (rs(it["les"].astype(np.uint8) * 255, Image.NEAREST) > 0).astype(np.uint8)
    leaf = (rs(it["leaf"].astype(np.uint8) * 255, Image.NEAREST) > 0).astype(np.uint8)
    return rgb, les, leaf


def augment(it, size, rng):
    img, les = it["img"], it["les"]
    H, W = les.shape
    s = rng.uniform(0.35, 1.0)                       # random zoom: crop side = s * short side
    side = int(min(H, W) * s)
    # bias half of the crops to contain lesion pixels
    if rng.random() < 0.5 and les.any():
        ys, xs = np.nonzero(les); j = rng.integers(len(ys))
        y0 = int(np.clip(ys[j] - rng.integers(0, side), 0, H - side)); x0 = int(np.clip(xs[j] - rng.integers(0, side), 0, W - side))
    else:
        y0 = rng.integers(0, H - side + 1); x0 = rng.integers(0, W - side + 1)
    ci = Image.fromarray(img[y0:y0 + side, x0:x0 + side]).resize((size, size), Image.BILINEAR)
    cm = Image.fromarray(les[y0:y0 + side, x0:x0 + side].astype(np.uint8) * 255).resize((size, size), Image.NEAREST)
    ci, cm = np.array(ci).astype(np.float32), (np.array(cm) > 0).astype(np.int64)
    if rng.random() < 0.5:
        ci, cm = ci[:, ::-1], cm[:, ::-1]
    if rng.random() < 0.5:
        ci, cm = ci[::-1], cm[::-1]
    ci = ci * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)            # brightness / contrast
    ci = (ci - ci.mean()) * rng.uniform(0.85, 1.15) + ci.mean()
    return np.clip(ci, 0, 255).astype(np.uint8).copy(), cm.copy()


# ------------------------------------------------------------------ training
def seg_loss(out, y):
    if out.shape[1] == 1:
        lo = out[:, 0]; yf = y.float()
        ce = F.binary_cross_entropy_with_logits(lo, yf); p = torch.sigmoid(lo)
    else:
        ce = F.cross_entropy(out, y); p = out.softmax(1)[:, 1]; yf = y.float()
    inter = (p * yf).sum((1, 2)); den = p.sum((1, 2)) + yf.sum((1, 2))
    dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
    return ce + dice


def logits(model, x):
    out = model(x)
    out = out[0] if isinstance(out, (tuple, list)) else out
    if out.shape[-1] != x.shape[-1]:
        out = F.interpolate(out, x.shape[-2:], mode="bilinear", align_corners=False)
    return out


def finetune(model, prep, train, a, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    # fixed optimisation budget (same as 40 photos x epochs), whatever the number of training photos
    steps = a.epochs * int(np.ceil(40 / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    order = []
    for _ in range(steps):
        if len(order) < min(a.bs, len(train)):
            order += list(rng.permutation(len(train)))
        idx, order = order[:min(a.bs, len(train))], order[min(a.bs, len(train)):]
        batch = [augment(train[i], a.size, rng) for i in idx]
        x = torch.cat([prep(im) for im, _ in batch])
        y = torch.from_numpy(np.stack([m for _, m in batch])).to(x.device)
        loss = seg_loss(logits(model, x), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    model.eval()
    return model


def evaluate(model, predict, test, size):
    rows = []
    for it in test:
        rgb, les, leaf = to_eval(it, size)
        pred = predict(model, rgb)
        if pred.shape != les.shape:
            pred = np.array(Image.fromarray(pred * 255).resize((size, size), Image.NEAREST)) > 0
        r = fe.image_row(pred.astype(np.uint8), les, leaf, fe.leaf_proxy_np(rgb))
        r.update(file=it["file"], disease=it["disease"]); rows.append(r)
    return pd.DataFrame(rows)


def imagenet_model(encoder, classes, dev):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet", classes=classes).to(dev)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=fe.EVAL)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--shots", type=int, nargs="*", default=None,
                    help="learning curve: labelled training photos PER DISEASE, e.g. --shots 2 5 10 (10 = all)")
    a = ap.parse_args()

    out = os.path.join(a.field, "results_curve" if a.shots else "results_ft"); os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    items = load_items(a.field)
    fold = folds_by_disease(items)
    print(f"photos with lesion + leaf labels: {len(items)}; per fold test sizes:",
          [fold.count(k) for k in range(3)])

    load, predict = fe.make_loader(a.work)
    lib = sys.modules.get("p1lib_rev") or sys.modules.get("p1lib")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def prep(rgb):
        if hasattr(lib, "to_tensor"):
            try:
                t = lib.to_tensor(rgb); t = t if t.dim() == 4 else t.unsqueeze(0)
                return t.to(dev).float()
            except Exception:
                pass
        x = (rgb.astype(np.float32) / 255 - fe.MEAN) / fe.STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(dev)

    mb, l0 = "mobilenetv3_small_100", "tf_efficientnet_lite0"
    seeds = [0] if a.quick else [0, 1, 2]
    inits = [(mb, m) for m in (["A0_student", "R4_augOnly"] if a.quick else
                               ["A0_student", "R4_augOnly", "R6_augCWD", "A7_full", "B_Caug"])]
    if not a.quick:
        inits += [(l0, "A0_student"), (l0, "A7_full")]
    inits += [(mb, "ImageNet")]
    if not a.quick:
        inits += [(l0, "ImageNet")]
        if not a.no_teacher:
            inits += [("teacher-segformer_b4", "teacher")]

    if a.shots:
        inits = [(mb, m) for m in ("A0_student", "R4_augOnly", "A7_full", "ImageNet")]
    shot_list = a.shots or [None]
    n_classes = None
    t0 = time.time()
    for shots in shot_list:
     for arch, method in inits:
        for seed in ([0] if method == "teacher" else seeds):
            for k in range(3):
                tag = f"{arch}__{method}__s{seed}__f{k}" + (f"__n{shots}" if shots else "")
                f_after = os.path.join(out, "runs", tag + "__after.csv")
                f_before = os.path.join(out, "runs", tag + "__before.csv")
                if os.path.exists(f_after):
                    continue
                train = [it for it, f in zip(items, fold) if f != k]
                if shots:   # same subset for every init: depends only on (fold, seed, shots)
                    r = np.random.default_rng(1000 * k + seed); sub = []
                    for d in sorted({it["disease"] for it in train}):
                        idx = [i for i, it in enumerate(train) if it["disease"] == d]
                        sub += list(r.permutation(idx)[:shots])
                    train = [train[i] for i in sorted(sub)]
                test = [it for it, f in zip(items, fold) if f == k]
                if method == "ImageNet":
                    if n_classes is None:
                        ref = load(f"{arch}__A0_student__s0")
                        n_classes = logits(ref, prep(np.zeros((a.size, a.size, 3), np.uint8))).shape[1]
                        del ref
                    try:
                        enc = "tu-" + arch
                        torch.manual_seed(seed)      # same random decoder for every fold / shot setting
                        model = imagenet_model(enc, n_classes, dev)
                    except Exception as e:
                        print(f"[skip] {tag}: cannot build ImageNet model ({e})"); break
                else:
                    run = f"{arch}__{method}__s{seed}"
                    if not os.path.exists(os.path.join(a.work, "runs", run, "best.pt")):
                        print(f"[skip] {run}: no checkpoint"); break
                    model = load(run)
                model.eval()
                evaluate(model, predict, test, a.size).assign(stage="before").to_csv(f_before, index=False)
                if method == "teacher":
                    bs0, a.bs = a.bs, 4
                model = finetune(model, prep, train, a, seed)
                if method == "teacher":
                    a.bs = bs0
                evaluate(model, predict, test, a.size).assign(stage="after").to_csv(f_after, index=False)
                del model; torch.cuda.empty_cache() if dev == "cuda" else None
                print(f"  {tag} done  ({(time.time() - t0) / 60:.1f} min)")

    # ---------------------------------------------------------- aggregate
    recs = []
    for f in glob.glob(os.path.join(out, "runs", "*.csv")):
        b = os.path.basename(f)[:-4]
        parts = b.split("__")
        stage = parts[-1]; parts = parts[:-1]
        n = int(parts.pop()[1:]) if parts[-1].startswith("n") and parts[-1][1:].isdigit() else 10
        arch, method, s, k = parts
        d = pd.read_csv(f); d["arch"], d["init"], d["seed"], d["fold"], d["stage"] = arch, method, int(s[1:]), int(k[1:]), stage
        d["shots"] = n
        recs.append(d)
    if not recs:
        sys.exit("nothing finished")
    P = pd.concat(recs, ignore_index=True)
    P.to_csv(os.path.join(out, "ft_per_image.csv"), index=False)

    rows = []
    for (arch, init, shots, seed, stage), d in P.groupby(["arch", "init", "shots", "seed", "stage"]):
        if d.fold.nunique() < 3:
            continue
        s = fe.summarise(d, f"{arch}__{init}__s{seed}__{stage}")
        s.update(arch=arch, init=init, shots=shots, seed=seed, stage=stage); rows.append(s)
    R = pd.DataFrame(rows)
    keep = ["lesion_iou_pooled", "lesion_iou_mean", "inleaf_lesion_iou", "fp_outside_leaf_pct",
            "sev_mae_leafgt", "sev_ccc_leafgt", "sev_acc_leafgt", "sev_kappa_leafgt", "sev_mae_proxy", "sev_kappa_proxy"]
    keep = [c for c in keep if c in R]
    g = R.groupby(["arch", "init", "shots", "stage"])
    S = g[keep].mean().join(g[keep].std().add_suffix("_sd")).join(g.size().rename("n_seeds")).reset_index()
    S = S.sort_values(["stage", "shots", "inleaf_lesion_iou"], ascending=[True, True, False])
    S.to_csv(os.path.join(out, "ft_summary.csv"), index=False)

    D = (P[P.stage == "after"].groupby(["arch", "init", "shots", "disease"])
         [["inleaf_lesion_iou"]].mean().mul(100).unstack("disease").round(1))
    D.to_csv(os.path.join(out, "ft_by_disease.csv"))

    # paired tests on per-image in-leaf IoU (seed-averaged), after fine-tuning
    try:
        from scipy.stats import wilcoxon
        A0_ = P[P.stage == "after"].groupby(["arch", "init", "shots", "file"]).inleaf_lesion_iou.mean().reset_index()
        T = []
        for (arch, shots), A in A0_.groupby(["arch", "shots"]):
            for ref in ("ImageNet", "A0_student"):
                base = A[(A.arch == arch) & (A.init == ref)].set_index("file").inleaf_lesion_iou
                if base.empty:
                    continue
                for init in A[A.arch == arch].init.unique():
                    if init == ref:
                        continue
                    x = A[(A.arch == arch) & (A.init == init)].set_index("file").inleaf_lesion_iou
                    j = pd.concat([x, base], axis=1, keys=["m", "b"]).dropna()
                    if len(j) >= 10 and (j.m != j.b).any():
                        st, p = wilcoxon(j.m, j.b)
                        T.append(dict(arch=arch, shots=shots, init=init, vs=ref, n=len(j), delta=100 * (j.m - j.b).mean(), p=p))
        if T:
            T = pd.DataFrame(T).sort_values("p"); m = len(T)
            T["p_holm"] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * T.p.values))
            T.to_csv(os.path.join(out, "ft_paired.csv"), index=False)
            print("\npaired (after fine-tuning, in-leaf lesion IoU):\n", T.round(4).to_string(index=False))
    except ImportError:
        pass

    pd.set_option("display.width", 220)
    show = ["arch", "init", "shots", "stage", "n_seeds"] + [c for c in ["lesion_iou_pooled", "inleaf_lesion_iou", "inleaf_lesion_iou_sd",
                                                            "fp_outside_leaf_pct", "sev_mae_leafgt", "sev_kappa_leafgt",
                                                            "sev_mae_proxy", "sev_kappa_proxy"] if c in S]
    print("\n", S[show].round(2).to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
