r"""
REV-1 (reviewer request): which part of CPD makes the student robust to clutter?
Trains 4 new configurations x 3 seeds on PlantSeg with the MobileNetV3-Small student, evaluates them on the clean
and cluttered test sets, and writes a summary table with the planned comparisons.

All runs are like-for-like with the C-aug control (no CWD, no LSD; same clutter, same lambda and p):

                         target view    occlusion mask   attention term
  B_Caug  (exists)       cluttered      no               no
  R3_clutterMask  NEW    cluttered      yes              no
  R5_cleanNoMask  NEW    clean          no               no
  R1_cleanMask    NEW    clean          yes              no
  A8_CPDonly (exists)    clean          yes              yes      (= CPD)
  R4_augOnly      NEW    no teacher: clutter augmentation + segmentation loss only

  clean view effect : R5 vs B_Caug (no mask)   and   R1 vs R3 (with mask)
  mask effect       : R3 vs B_Caug (cluttered) and   R1 vs R5 (clean)
  attention effect  : A8 vs R1
  value of KD       : A8 vs R4

Uses p1lib/p1lib_rev.py (p1lib + the factorial methods; found automatically).
Do not run the notebook at the same time (one GPU).

  conda activate cropai
  cd <repo>/scripts
  python rev1_ablation.py run          # ~1-2 days; safe to stop (Ctrl+C) and start again: runs resume where they were
  python rev1_ablation.py status       # what is done
  python rev1_ablation.py summary      # table -> <work>\results\table_rev1.csv

Optional: python rev1_ablation.py run "<other work folder>"
"""
import sys, time, traceback
import os
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "p1lib"))
import p1lib_rev as L

WORK = Path(os.environ.get("P1_ROOT", ".")) / r"paper1_runs_v2"
NEW = ["R1_cleanMask", "R3_clutterMask", "R5_cleanNoMask", "R4_augOnly", "R6_augCWD"]
SEEDS = [0, 1, 2]
TESTS = ["plantseg_test", "clutter_light", "clutter_medium", "clutter_heavy"]
SHOW = ["A0_student", "R4_augOnly", "R6_augCWD", "B_Caug", "R3_clutterMask", "R5_cleanNoMask", "R1_cleanMask", "A8_CPDonly", "A6_CPD", "A7_full"]
CONTRASTS = [("clean view, no mask", "R5_cleanNoMask", "B_Caug"), ("clean view, with mask", "R1_cleanMask", "R3_clutterMask"),
             ("mask, cluttered target", "R3_clutterMask", "B_Caug"), ("mask, clean target", "R1_cleanMask", "R5_cleanNoMask"),
             ("attention term", "A8_CPDonly", "R1_cleanMask"), ("CPD vs augmentation only", "A8_CPDonly", "R4_augOnly"),
             ("CPD vs C-aug", "A8_CPDonly", "B_Caug"),
             ("CPD on top of aug+CWD", "A6_CPD", "R6_augCWD"), ("full method vs aug+CWD", "A7_full", "R6_augCWD"),
             ("CWD on top of augmentation", "R6_augCWD", "R4_augOnly")]


def setup(work):
    cfg = L.load_cfg(work); cfg["paths"]["work"] = str(work)
    if cfg.get("method_set") != "v2":
        sys.exit(f"{work}\\config.json has method_set={cfg.get('method_set')!r}; expected 'v2' (the PlantSeg run folder)")
    if cfg.get("quick"):
        sys.exit("config.json has quick=true; this is the pilot folder, not the full run folder")
    return cfg


def jobs(cfg):
    st = cfg["students"][0]
    return [dict(name=f"{L.short(st)}__{m}__s{sd}", kind="student", encoder=st, arch="Unet", method=m, seed=sd)
            for sd in SEEDS for m in NEW]            # finished runs are skipped, so only R6 / R7 train now


def run(work):
    import torch
    cfg = setup(work)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if hasattr(L, "keep_awake") and L.keep_awake():
        L.log("Windows sleep blocked while training runs")
    frames = L.load_split_frames(cfg); bank = L.make_bank(frames["train"], cfg)
    L.log(f"device {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU (slow!)'} | train {len(frames['train'])} "
          f"val {len(frames['val'])} test {len(frames['test'])}")
    for j in jobs(cfg):
        if L.run_done(cfg, j["name"]) and L.eval_done(cfg, j["name"]):
            L.log(f"already done: {j['name']}"); continue
        t0 = time.time(); L.log(f"START {j['name']}")
        try:
            L.train_run(cfg, j, frames, bank, dev, None)
            L.eval_run(cfg, j["name"], frames, dev)
        except Exception:
            L.wp(cfg, "runs", j["name"]).mkdir(parents=True, exist_ok=True)
            L.wp(cfg, "runs", j["name"], "error.txt").write_text(traceback.format_exc())
            raise
        L.log(f"DONE {j['name']} in {(time.time() - t0) / 3600:.1f} h")
    L.log("all REV-1 runs finished -> python rev1_ablation.py summary")


def status(work):
    cfg = setup(work)
    for j in jobs(cfg):
        rd = L.wp(cfg, "runs", j["name"])
        st = "done+evaluated" if L.eval_done(cfg, j["name"]) else "trained" if L.run_done(cfg, j["name"]) else \
             "ERROR (see error.txt)" if (rd / "error.txt").exists() else "in progress" if (rd / "last.pt").exists() else "not started"
        print(f"{j['name']:45s} {st}")


def per_run(pi, run, test):
    p = pi / f"{run}__{test}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, dtype={"key": str}); s = lambda c: d[c].sum()
    return dict(miou=100 * (s("inter1") / max(s("union1"), 1) + s("inter0") / max(s("union0"), 1)) / 2,
                iou_lesion=100 * s("inter1") / max(s("union1"), 1))


def summary(work):
    from scipy.stats import ttest_ind, wilcoxon
    cfg = setup(work); pi = L.wp(cfg, "results", "per_image"); stu = L.short(cfg["students"][0])
    rows = []
    for m in SHOW:
        for sd in SEEDS:
            for t in TESTS:
                r = per_run(pi, f"{stu}__{m}__s{sd}", t)
                if r:
                    rows.append(dict(method=m, seed=sd, test=t, **r))
    R = pd.DataFrame(rows)
    if R.empty:
        sys.exit("no results found yet")
    tab = R.pivot_table(index=["method", "seed"], columns="test", values="miou").reset_index()
    for t in TESTS:
        if t not in tab:
            tab[t] = np.nan
    tab["drop_heavy"] = tab["plantseg_test"] - tab["clutter_heavy"]
    agg = tab.groupby("method").agg(["mean", "std", "count"])
    out = []
    for m in SHOW:
        if m not in agg.index:
            continue
        a = agg.loc[m]
        out.append(dict(method=m, seeds=int(a[("plantseg_test", "count")]),
                        **{f"{c}": f"{a[(c, 'mean')]:.2f} ± {0 if np.isnan(a[(c, 'std')]) else a[(c, 'std')]:.2f}"
                           for c in ["plantseg_test", "clutter_medium", "clutter_heavy", "drop_heavy"] if (c, "mean") in a}))
    T = pd.DataFrame(out)
    print("\nmIoU (%), mean ± s.d. over seeds, MobileNetV3-Small, PlantSeg\n", T.to_string(index=False))

    def img_iou(m, t):                                   # per-image lesion IoU averaged over seeds
        ds = [pd.read_csv(pi / f"{stu}__{m}__s{sd}__{t}.csv", dtype={"key": str}).set_index("key").iou
              for sd in SEEDS if (pi / f"{stu}__{m}__s{sd}__{t}.csv").exists()]
        return pd.concat(ds, axis=1).mean(axis=1) if ds else None

    cres = []
    for name, a, b in CONTRASTS:
        for t in ["plantseg_test", "clutter_heavy"]:
            xa, xb = tab[tab.method == a][t].dropna(), tab[tab.method == b][t].dropna()
            ia, ib = img_iou(a, t), img_iou(b, t)
            if len(xa) < 2 or len(xb) < 2 or ia is None or ib is None:
                continue
            p = ttest_ind(xa, xb, equal_var=False).pvalue
            d = (ia - ib.loc[ia.index]).to_numpy(); rng = np.random.default_rng(0)
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
            cres.append(dict(contrast=name, a=a, b=b, test=t, miou_diff=round(xa.mean() - xb.mean(), 2), welch_p=round(p, 4),
                             img_lesion_iou_gain=round(100 * d.mean(), 2), ci_low=round(100 * np.percentile(bs, 2.5), 2),
                             ci_high=round(100 * np.percentile(bs, 97.5), 2),
                             wilcoxon_p=round(wilcoxon(d).pvalue, 4) if np.any(d != 0) else 1.0))
    C = pd.DataFrame(cres)
    if len(C):
        print("\nPlanned comparisons (a minus b)\n", C.to_string(index=False))
    res = L.wp(cfg, "results"); res.mkdir(exist_ok=True)
    T.to_csv(res / "table_rev1.csv", index=False); C.to_csv(res / "table_rev1_contrasts.csv", index=False)
    tab.to_csv(res / "table_rev1_per_seed.csv", index=False)
    print("\nsaved", res / "table_rev1.csv", "+ table_rev1_contrasts.csv + table_rev1_per_seed.csv")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    work = Path(sys.argv[2]) if len(sys.argv) > 2 else WORK
    {"run": run, "status": status, "summary": summary}.get(cmd, lambda w: print(__doc__))(work)
