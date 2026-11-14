"""
field_eval.py  -  Paper 1, real-field test on the 60 rice photographs.

Runs every trained model in a run folder (WORK) on the labelled field photos and
writes paper-ready tables. No training, CPU is fine (GPU used if available).

Folder layout expected (FIELD):
    FIELD/images/<stem>.jpg            the 60 photos from field_test.zip
    FIELD/labels/<stem>.json           LabelMe file, polygons labelled "lesion" and "leaf"
        or
    FIELD/labels/<stem>_lesion.png     binary mask, nonzero = lesion
    FIELD/labels/<stem>_leaf.png       binary mask, nonzero = target leaf (optional but wanted)
    FIELD/field_images.csv             file, disease, original   (comes with the zip)
Images without a label file are skipped (and listed in the log).

Usage (Windows, conda env cropai, in <repo>):
    python field_eval.py --work "<P1_ROOT>\\paper1_runs_v2" --field "<P1_ROOT>\\field_test"
    python field_eval.py ... --runs A0 R4 R6 A7 teacher      # only some methods
    python field_eval.py --demo                                # smoke test, no models needed

Outputs in FIELD/results/:
    per_image_<run>.csv           one row per photo per run
    field_runs.csv                one row per run (all metrics)
    field_methods.csv             mean +- sd over seeds per method  <- Table for the paper
    field_by_disease.csv          lesion IoU per method x disease
    field_paired.csv              paired Wilcoxon vs A0 on per-image lesion IoU (seed-averaged)
    field_table.tex               LaTeX rows for the paper
"""
import argparse, glob, json, math, os, sys, re
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

EVAL = 320                                    # same as training / clutter evaluation
CLASS_EDGES = (5.0, 25.0)                     # severity classes: <5 | 5-25 | >25 % of leaf
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ----------------------------------------------------------------- labels
def load_labels(lab_dir, stem, size):
    """Return (lesion, leaf) uint8 masks at the original image size, or (None, None)."""
    W, H = size
    js = os.path.join(lab_dir, stem + ".json")
    if os.path.exists(js):
        d = json.load(open(js, encoding="utf-8"))
        sx = W / d.get("imageWidth", W); sy = H / d.get("imageHeight", H)
        les = Image.new("L", (W, H), 0); leaf = Image.new("L", (W, H), 0)
        dl, df = ImageDraw.Draw(les), ImageDraw.Draw(leaf)
        n_leaf = 0
        for s in d.get("shapes", []):
            lab = s.get("label", "").strip().lower()
            pts = [(x * sx, y * sy) for x, y in s.get("points", [])]
            if len(pts) < 3:
                continue
            if lab.startswith("lesion"):
                dl.polygon(pts, fill=1)
            elif lab.startswith("leaf"):
                df.polygon(pts, fill=1); n_leaf += 1
        les = np.array(les, np.uint8); leaf = np.array(leaf, np.uint8)
        return les, (leaf if n_leaf else None)
    p_les = os.path.join(lab_dir, stem + "_lesion.png")
    if os.path.exists(p_les):
        les = (np.array(Image.open(p_les).convert("L").resize((W, H), Image.NEAREST)) > 0).astype(np.uint8)
        p_leaf = os.path.join(lab_dir, stem + "_leaf.png")
        leaf = None
        if os.path.exists(p_leaf):
            leaf = (np.array(Image.open(p_leaf).convert("L").resize((W, H), Image.NEAREST)) > 0).astype(np.uint8)
        return les, leaf
    return None, None


def leaf_proxy_np(rgb):
    """Excess-green + Otsu (same rule as p1lib.leaf_proxy and the phone app)."""
    x = rgb.astype(np.float32)
    exg = 2 * x[..., 1] - x[..., 0] - x[..., 2]
    v = exg.ravel()
    hist, edges = np.histogram(v, bins=256)
    p = hist / max(hist.sum(), 1); w = np.cumsum(p); m = np.cumsum(p * edges[:-1])
    mt = m[-1]; sb = (mt * w - m) ** 2 / np.maximum(w * (1 - w), 1e-12)
    t = edges[int(np.nanargmax(sb))]
    return (exg > t).astype(np.uint8)


# ----------------------------------------------------------------- metrics
def ccc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return float("nan")
    ma, mb = a.mean(), b.mean(); va, vb = a.var(), b.var(); cov = ((a - ma) * (b - mb)).mean()
    return 2 * cov / (va + vb + (ma - mb) ** 2 + 1e-12)


def sev_class(s):
    return np.digitize(np.asarray(s, float), CLASS_EDGES)


def qw_kappa(a, b, k=3):
    a, b = np.asarray(a, int), np.asarray(b, int)
    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i, j] += 1
    if O.sum() == 0:
        return float("nan")
    Wt = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (Wt * E).sum()
    return 1 - (Wt * O).sum() / den if den > 0 else float("nan")


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    r = np.random.default_rng(seed)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return tuple(np.percentile(m, [2.5, 97.5]))


def image_row(pred, les, leaf, proxy):
    """All per-image quantities at EVAL resolution. pred/les/leaf/proxy are 0/1 arrays."""
    r = {}
    i1 = int((pred & les).sum()); u1 = int((pred | les).sum())
    i0 = int(((1 - pred) & (1 - les)).sum()); u0 = int(((1 - pred) | (1 - les)).sum())
    r.update(inter1=i1, union1=u1, inter0=i0, union0=u0)
    r["lesion_iou"] = i1 / u1 if u1 else float("nan")          # nan when no lesion anywhere
    r["miou"] = np.nanmean([r["lesion_iou"], i0 / u0 if u0 else np.nan])
    r["true_frac_img"] = 100 * les.mean(); r["pred_frac_img"] = 100 * pred.mean()
    # false positives outside the target leaf (clutter firing as lesion)
    if leaf is not None:
        out = (1 - leaf)
        r["fp_outside_leaf_pct"] = 100 * (pred & out).sum() / max(out.sum(), 1)
        L = leaf.sum()
        tl = les & leaf; pl = pred & leaf
        r["leaf_px"] = int(L)
        r["inleaf_lesion_iou"] = (tl & pl).sum() / max((tl | pl).sum(), 1) if (tl | pl).sum() else float("nan")
        r["sev_true"] = 100 * tl.sum() / max(L, 1)                 # ground truth severity
        r["sev_pred_leafgt"] = 100 * pl.sum() / max(L, 1)          # model, with the true leaf given
        # what the phone app reports: lesion / (proxy U lesion), on the whole photo
        r["sev_pred_proxy"] = 100 * pred.sum() / max((proxy | pred).sum(), 1)
        r["proxy_leaf_ratio"] = (proxy | pred).sum() / max(L, 1)   # >1 means proxy grabs grass/weeds
    return r


def summarise(df, run):
    s = {"run": run, "n_img": len(df)}
    s["lesion_iou_pooled"] = 100 * df.inter1.sum() / max(df.union1.sum(), 1)
    s["miou_pooled"] = 100 * 0.5 * (df.inter1.sum() / max(df.union1.sum(), 1) + df.inter0.sum() / max(df.union0.sum(), 1))
    s["lesion_iou_mean"] = 100 * df.lesion_iou.mean()
    lo, hi = boot_ci(100 * df.lesion_iou)
    s["lesion_iou_ci_lo"], s["lesion_iou_ci_hi"] = lo, hi
    if "sev_true" in df and df.sev_true.notna().any():
        d = df.dropna(subset=["sev_true"])
        s["n_sev"] = len(d)
        s["fp_outside_leaf_pct"] = d.fp_outside_leaf_pct.mean()
        s["inleaf_lesion_iou"] = 100 * d.inleaf_lesion_iou.mean()
        for tag in ("leafgt", "proxy"):
            p = d["sev_pred_" + tag]
            s[f"sev_mae_{tag}"] = (p - d.sev_true).abs().mean()
            s[f"sev_bias_{tag}"] = (p - d.sev_true).mean()
            s[f"sev_ccc_{tag}"] = ccc(d.sev_true, p)
            ct, cp = sev_class(d.sev_true), sev_class(p)
            s[f"sev_acc_{tag}"] = 100 * (ct == cp).mean()
            s[f"sev_kappa_{tag}"] = qw_kappa(ct, cp)
            diff = p - d.sev_true
            s[f"sev_loa_{tag}"] = 1.96 * diff.std(ddof=1) if len(d) > 1 else float("nan")
        s["proxy_leaf_ratio_median"] = d.proxy_leaf_ratio.median()
    return s


# ----------------------------------------------------------------- models
def parse_run(name):
    m = re.match(r"^(.*?)__(.*?)__s(\d+)$", name)
    return (m.group(1), m.group(2), int(m.group(3))) if m else (name, name, 0)


def short_method(method):
    for k in ("A0", "A5", "A6", "A7", "A8", "R1", "R3", "R4", "R5", "R6", "B_", "C"):
        if method.startswith(k):
            return method
    return method


def make_loader(work):
    """Use the project's own loader so weights and preprocessing match training exactly."""
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "p1lib"))
    lib = None
    for name in ("p1lib_rev", "p1lib"):
        try:
            lib = __import__(name); break
        except Exception as e:  # noqa
            print(f"[warn] cannot import {name}: {e}")
    if lib is None:
        sys.exit("Run this from the folder that holds p1lib_rev.py / p1lib.py (<repo>).")
    cfg = lib.load_cfg(work) if hasattr(lib, "load_cfg") else {}
    import inspect, torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def load(run):
        lm = lib.load_model
        params = list(inspect.signature(lm).parameters)
        # p1lib signature is load_model(<cfg or work>, name, dev)
        tries = [lambda: lm(cfg, run, dev), lambda: lm(work, run, dev),
                 lambda: lm(cfg, run, torch.device(dev)), lambda: lm(work, run, torch.device(dev)),
                 lambda: lm(cfg, run), lambda: lm(work, run)]
        errs = []
        for t in tries:
            try:
                out = t()
                model = out[0] if isinstance(out, (tuple, list)) else out
                return model.to(dev).eval()
            except Exception as e:  # noqa
                errs.append(f"{type(e).__name__}: {e}")
        raise RuntimeError(f"load_model{tuple(params)} failed for {run}: " + " | ".join(errs[:2]))

    to_t = getattr(lib, "to_tensor", None)

    def prep(rgb320):
        if to_t is not None:
            try:
                t = to_t(rgb320)
                t = t if t.dim() == 4 else t.unsqueeze(0)
                return t.to(dev)
            except Exception:
                pass
        x = (rgb320.astype(np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(dev)

    @torch.no_grad()
    def predict(model, rgb320):
        out = model(prep(rgb320))
        out = out[0] if isinstance(out, (tuple, list)) else out
        if out.shape[-1] != EVAL:
            out = torch.nn.functional.interpolate(out, (EVAL, EVAL), mode="bilinear", align_corners=False)
        if out.shape[1] == 1:
            return (out[0, 0] > 0).cpu().numpy().astype(np.uint8)
        return out[0].argmax(0).cpu().numpy().astype(np.uint8)

    return load, predict


def demo_loader():
    """Fake models for a smoke test: thresholds on brown/yellow pixels with different strengths."""
    def load(run):
        return {"A0__s0": 0.30, "A0__s1": 0.32, "R4_augOnly__s0": 0.45, "R4_augOnly__s1": 0.47,
                "teacher__s0": 0.55}[run.split("__", 1)[1]]

    def predict(k, rgb):
        x = rgb.astype(np.float32) / 255
        brown = (x[..., 0] > x[..., 1] * 0.9) & (x[..., 2] < 0.45)
        rng = np.random.default_rng(int(k * 100))
        noise = rng.random(brown.shape) > (1 - 0.05 / k)
        return (brown | noise).astype(np.uint8)
    return load, predict


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", help="run folder, e.g. <P1_ROOT>\\paper1_runs_v2")
    ap.add_argument("--field", default="field_test")
    ap.add_argument("--runs", nargs="*", default=None, help="method name filters, e.g. A0 R4 A7 teacher")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--crop", action="store_true",
                    help="crop each photo to a square around the labelled leaf (as if the farmer framed the leaf)")
    ap.add_argument("--sanity", type=int, default=0,
                    help="check preprocessing: run models on N PlantSeg test images and print lesion IoU")
    a = ap.parse_args()
    if a.sanity:
        return sanity(a)

    img_dir = os.path.join(a.field, "images"); lab_dir = os.path.join(a.field, "labels")
    out_dir = os.path.join(a.field, "results_crop" if a.crop else "results"); os.makedirs(out_dir, exist_ok=True)
    meta = pd.read_csv(os.path.join(a.field, "field_images.csv")) if os.path.exists(
        os.path.join(a.field, "field_images.csv")) else None
    disease = dict(zip(meta.file, meta.disease)) if meta is not None else {}

    # load images + labels once
    items, skipped = [], []
    for p in sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        im = Image.open(p).convert("RGB")
        les, leaf = load_labels(lab_dir, stem, im.size)
        if les is None:
            skipped.append(stem); continue
        if a.crop and leaf is not None:
            ys, xs = np.nonzero(leaf); W0, H0 = im.size
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            half = max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.575 + 16
            half = min(half, max(W0, H0) / 2)
            x0 = int(np.clip(cx - half, 0, max(W0 - 2 * half, 0))); y0 = int(np.clip(cy - half, 0, max(H0 - 2 * half, 0)))
            box = (x0, y0, int(min(x0 + 2 * half, W0)), int(min(y0 + 2 * half, H0)))
            im = im.crop(box); les = les[box[1]:box[3], box[0]:box[2]]; leaf = leaf[box[1]:box[3], box[0]:box[2]]
        rgb = np.array(im.resize((EVAL, EVAL), Image.BILINEAR))
        les = np.array(Image.fromarray(les * 255).resize((EVAL, EVAL), Image.NEAREST)) > 0
        leaf = (np.array(Image.fromarray(leaf * 255).resize((EVAL, EVAL), Image.NEAREST)) > 0) if leaf is not None else None
        items.append(dict(stem=stem, file=os.path.basename(p), rgb=rgb, les=les.astype(np.uint8),
                          leaf=None if leaf is None else leaf.astype(np.uint8), proxy=leaf_proxy_np(rgb),
                          disease=disease.get(os.path.basename(p), stem.rsplit("_", 1)[0])))
    print(f"labelled images: {len(items)}   skipped (no label): {len(skipped)}")
    if skipped:
        print("  skipped:", ", ".join(skipped[:20]), "..." if len(skipped) > 20 else "")
    if not items:
        sys.exit("No labelled images found.")
    n_noleaf = sum(it["leaf"] is None for it in items)
    if n_noleaf:
        print(f"[warn] {n_noleaf} images have no 'leaf' polygon: severity columns skipped for them")

    if a.demo:
        load, predict = demo_loader()
        runs = ["x__A0__s0", "x__A0__s1", "x__R4_augOnly__s0", "x__R4_augOnly__s1", "t__teacher__s0"]
    else:
        if not a.work:
            sys.exit("--work is required (or use --demo)")
        load, predict = make_loader(a.work)
        runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(a.work, "runs", "*", "best.pt")))
        if a.runs:
            runs = [r for r in runs if any(f.lower() in r.lower() for f in a.runs)]
    print(f"runs: {len(runs)}")

    rows = []
    for run in runs:
        f = os.path.join(out_dir, f"per_image_{run}.csv")
        if os.path.exists(f) and not a.demo:
            df = pd.read_csv(f); print(f"  {run}: cached")
        else:
            try:
                model = load(run)
            except Exception as e:
                print(f"  [skip] {run}: {e}"); continue
            recs = []
            for it in items:
                pred = predict(model, it["rgb"])
                r = image_row(pred, it["les"], it["leaf"], it["proxy"])
                r.update(file=it["file"], disease=it["disease"])
                recs.append(r)
            df = pd.DataFrame(recs); df.to_csv(f, index=False)
            print(f"  {run}: lesion IoU {100 * df.lesion_iou.mean():.1f}")
        arch, method, seed = parse_run(run)
        s = summarise(df, run); s.update(arch=arch, method=method, seed=seed)
        rows.append(s)
        df["run"], df["arch"], df["method"], df["seed"] = run, arch, method, seed
        rows[-1]["_df"] = df

    if not rows:
        sys.exit("no run evaluated")
    per = pd.concat([r.pop("_df") for r in rows], ignore_index=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(out_dir, "field_runs.csv"), index=False)

    num = [c for c in R.columns if c not in ("run", "arch", "method", "seed") and pd.api.types.is_numeric_dtype(R[c])]
    g = R.groupby(["arch", "method"])
    M = g[num].mean().add_suffix("")
    S = g[num].std().add_suffix("_sd")
    M = M.join(S).join(g.size().rename("n_seeds")).reset_index()
    M.to_csv(os.path.join(out_dir, "field_methods.csv"), index=False)

    # per disease
    D = (per.groupby(["arch", "method", "seed", "disease"])
            .apply(lambda d: 100 * d.inter1.sum() / max(d.union1.sum(), 1)).rename("lesion_iou_pooled").reset_index()
            .groupby(["arch", "method", "disease"]).lesion_iou_pooled.mean().unstack("disease").reset_index())
    D.to_csv(os.path.join(out_dir, "field_by_disease.csv"), index=False)

    # paired test vs A0 (same student), on seed-averaged per-image lesion IoU
    try:
        from scipy.stats import wilcoxon
        P = []
        avg = per.groupby(["arch", "method", "file"]).lesion_iou.mean().reset_index()
        for arch in avg.arch.unique():
            base = avg[(avg.arch == arch) & (avg.method.str.startswith("A0"))].set_index("file").lesion_iou
            if base.empty:
                continue
            for m in avg[avg.arch == arch].method.unique():
                if m.startswith("A0"):
                    continue
                x = avg[(avg.arch == arch) & (avg.method == m)].set_index("file").lesion_iou
                j = pd.concat([x, base], axis=1, keys=["m", "b"]).dropna()
                if len(j) < 5:
                    continue
                st, p = wilcoxon(j.m, j.b)
                P.append(dict(arch=arch, method=m, n=len(j), delta=100 * (j.m - j.b).mean(), p=p))
        if P:
            P = pd.DataFrame(P).sort_values("p")
            k = len(P); P["p_holm"] = np.minimum(1, np.maximum.accumulate((k - np.arange(k)) * P.p.values))
            P.to_csv(os.path.join(out_dir, "field_paired.csv"), index=False)
    except ImportError:
        print("[warn] scipy missing: paired tests skipped")

    # LaTeX rows
    def pm(m, s):
        return f"{m:.1f}" if (s != s or s == 0) else f"{m:.1f}$\\pm${s:.1f}"
    lines = []
    for _, r in M.iterrows():
        cells = [r.method.replace("_", r"\_"), pm(r.lesion_iou_pooled, r.get("lesion_iou_pooled_sd", np.nan))]
        if "sev_mae_leafgt" in M:
            cells += [pm(r.fp_outside_leaf_pct, r.fp_outside_leaf_pct_sd),
                      f"{r.sev_mae_leafgt:.1f}", f"{r.sev_ccc_leafgt:.2f}", f"{r.sev_kappa_leafgt:.2f}",
                      f"{r.sev_mae_proxy:.1f}", f"{r.sev_kappa_proxy:.2f}"]
        lines.append(" & ".join(cells) + r" \\")
    hdr = "% method & lesion IoU & FP outside leaf % & MAE(leaf) & CCC(leaf) & kappa(leaf) & MAE(proxy) & kappa(proxy)\n"
    open(os.path.join(out_dir, "field_table.tex"), "w").write(hdr + "\n".join(lines) + "\n")

    show = [c for c in ["arch", "method", "n_seeds", "lesion_iou_pooled", "lesion_iou_mean", "inleaf_lesion_iou", "fp_outside_leaf_pct",
                        "sev_mae_leafgt", "sev_kappa_leafgt", "sev_mae_proxy", "sev_kappa_proxy", "proxy_leaf_ratio_median"] if c in M]
    pd.set_option("display.width", 200)
    print(M[show].round(2).to_string(index=False))
    print(f"\nwritten to {out_dir}")


def sanity(a):
    """Same loader + preprocessing on PlantSeg test images. If these numbers are far below the paper's
    clean-test lesion IoU, the field numbers are a preprocessing bug, not a real result."""
    sp = pd.read_csv(os.path.join(a.work, "splits", "split.csv"))
    sp = sp[sp.split == "test"].head(a.sanity)
    caches = sorted(glob.glob(os.path.join(a.work, "cache_*")))
    if not caches:
        sys.exit("no cache_* folder in WORK")
    cdir = caches[-1]
    def index(d):
        return {os.path.splitext(f)[0]: os.path.join(d, f) for f in os.listdir(d)} if os.path.isdir(d) else {}
    I_, M_ = index(os.path.join(cdir, "img")), index(os.path.join(cdir, "msk"))
    imgs = []
    for _, r in sp.iterrows():
        cands = [str(r.uid), os.path.splitext(str(r.uid))[0]]
        for c in ("image", "mask"):
            if c in r:
                cands.append(os.path.splitext(os.path.basename(str(r[c])))[0])
        k = next((c for c in cands if c in I_ and c in M_), None)
        if k is None:
            continue
        rgb = np.array(Image.open(I_[k]).convert("RGB").resize((EVAL, EVAL), Image.BILINEAR))
        m = (np.array(Image.open(M_[k]).resize((EVAL, EVAL), Image.NEAREST)) > 0).astype(np.uint8)
        imgs.append((rgb, m))
    if not imgs:
        print("[diag] split.csv columns:", list(sp.columns))
        print("[diag] first test rows:", sp.head(2).to_dict("records"))
        print("[diag] cache folders:", caches, " img files:", list(I_)[:3], " msk files:", list(M_)[:3])
        # fallback: the frozen clutter test set at level 'light' has images + masks side by side
        for lvl in ("light", "medium", "heavy"):
            d = os.path.join(a.work, "clutter_test", lvl)
            Ii, Mm = index(os.path.join(d, "images")), index(os.path.join(d, "masks"))
            for k in list(Ii)[:a.sanity]:
                if k in Mm:
                    rgb = np.array(Image.open(Ii[k]).convert("RGB").resize((EVAL, EVAL), Image.BILINEAR))
                    m = (np.array(Image.open(Mm[k]).resize((EVAL, EVAL), Image.NEAREST)) > 0).astype(np.uint8)
                    imgs.append((rgb, m))
            if imgs:
                print(f"[diag] using clutter_test/{lvl} instead (compare with the paper's {lvl}-clutter lesion IoU)")
                break
    print(f"sanity: {len(imgs)} PlantSeg test images from {cdir}")
    load, predict = make_loader(a.work)
    runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(a.work, "runs", "*", "best.pt")))
    keep = a.runs or ["A0_student__s0", "R4_augOnly__s0", "A7_full__s0", "teacher"]
    runs = [r for r in runs if any(f.lower() in r.lower() for f in keep)]
    for run in runs:
        model = load(run); I = U = 0
        for rgb, m in imgs:
            p = predict(model, rgb); I += (p & m).sum(); U += (p | m).sum()
        print(f"  {run}: PlantSeg-test lesion IoU (pooled) {100 * I / max(U, 1):.1f}")


if __name__ == "__main__":
    main()
