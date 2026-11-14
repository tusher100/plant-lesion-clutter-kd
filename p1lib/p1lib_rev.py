"""
p1lib.py - Lesion-Aware, Clutter-Robust Distillation (Paper 1): full experiment pipeline.

Follows "Paper 1 Methodology" sections 3-5 and 7:
  data      leak-free 70/15/15 split stratified by disease (pHash groups never span splits), cached images
  clutter   leaf-like occluders, shadows, dirt, glare, background swap, blur, JPEG; altered share tracked
  models    EfficientNet-B4 U-Net teacher; MobileNetV3-Small / EfficientNet-Lite0 U-Net students;
            direct small baselines SegFormer-B0 and DeepLabV3+ MobileNetV2 (no distillation)
  losses    LSD, CPD, logit KD, AT, CWD, SKD (pixel + pair-wise), BPKD (body + edge), SARD (re-implemented), C-aug
  runner    resumable job queue with the week-3 and week-7 gates, time budget per session, one worker per GPU
  eval      mIoU, Dice, lesion F1, recall by lesion size, boundary F1, severity error, per disease, clutter, natural clutter

Run as a worker:  python p1lib.py worker <gpu_index>   (the notebook launches these)
"""
import os, sys, json, math, time, random, collections, shutil, glob, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, cv2

cv2.setNumThreads(0)
import torch, torch.nn as nn, torch.nn.functional as F

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

DEFAULT_CFG = dict(
    img_size=320, cache_long_side=640,               # cache = 2x train size (enough for random scale up to 2.0)
    # teacher candidates; the pilot trains both and "auto" keeps the one with the higher val mIoU
    teachers={"segformer_b4": dict(arch="Segformer", encoder="mit_b4", transformer=True),
              "resnet50": dict(arch="Unet", encoder="resnet50", transformer=False)},
    teachers_to_train=["segformer_b4", "resnet50"], teacher_choice="auto",
    students=["tu-mobilenetv3_small_100"],              # primary student: every method, ablation and sensitivity run
    secondary_students=["tu-tf_efficientnet_lite0"],    # second student: only A0, best baseline and A7 (shows generality)
    secondary_methods=["A0_student", "best_baseline", "A7_full"],
    direct=[dict(name="D-segformer_b0", arch="Segformer", encoder="mit_b0", transformer=True),
            dict(name="D-dlv3p_mobilenetv2", arch="DeepLabV3Plus", encoder="mobilenet_v2", transformer=False)],
    teacher_epochs=40, student_epochs=40, teacher_bs=8, student_bs=16,
    val_every=2, patience_evals=4,                   # validate every 2 epochs; stop a run after 4 checks (8 epochs) without improvement
    lr_teacher=3e-4, lr_student=1e-3, lr_transformer=6e-5, weight_decay=1e-2, warmup_iters=1500,
    seeds=[0, 1, 2], final_seeds=[],
    kd=dict(tau=4.0, lambda_lsd=1.0, lambda_cpd=1.0, mu_feat=1.0, rho_att=1.0, cpd_kl=1.0,
            alpha_boundary=2.0, beta_size=1.0, gamma_conf=1.0, delta_leaf=0.3,
            feat_stages=[3, 4, 5], clutter_prob=0.5, boundary_k=5, skd_w=1.0),
    size_bins=[0.005, 0.05],          # lesion component area / image area: small < 0.5% <= medium <= 5% < large
    split=[0.70, 0.15, 0.15], phash_thr=6,
    gates=dict(teacher_margin=0.05, full_margin=0.015, stop_on_fail=True),
    pretrained=True, num_workers=2, eval_bs=16,
    quick=False, quick_n=[64, 32, 32], quick_epochs=2, max_phase=99,
    sensitivity=["S_tau1", "S_tau2", "S_lsd0.5", "S_lsd2", "S_cpd0.5", "S_cpd2"],
    time_budget_h=11.0, deadline=None,
    ram_high=88, ram_critical=95,             # % of system RAM: >high halves the data workers next epoch, >critical pauses briefly
    gpu_temp_high=85, gpu_temp_resume=75,     # deg C: pause training at >= high, resume when <= resume (max 30 min per pause)
)

# --------------------------------------------------------------------------------------- config & paths
def load_cfg(work):
    with open(Path(work) / "config.json") as f:
        return json.load(f)


def save_cfg(cfg, work):
    Path(work).mkdir(parents=True, exist_ok=True)
    atomic_write_text(Path(work) / "config.json", json.dumps(cfg, indent=2))


def wp(cfg, *parts):
    return Path(cfg["paths"]["work"]).joinpath(*parts)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def short(enc):
    return enc.replace("tu-", "").split("/")[-1]


def parse_run(name):
    model, method, seed = name.split("__")
    return model, method, int(seed[1:])


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)

def atomic_write_text(path, text):
    """Write to a temporary file, then rename: a crash or power cut never leaves a half-written file."""
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)

def atomic_to_csv(df, path, **kw):
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, **kw); os.replace(tmp, path)

_NVSMI = {"path": None, "checked": False}

def gpu_temp():
    """Hottest GPU temperature in deg C from nvidia-smi, or None if it cannot be read."""
    if not _NVSMI["checked"]:
        _NVSMI["path"] = shutil.which("nvidia-smi"); _NVSMI["checked"] = True
    if not _NVSMI["path"]:
        return None
    try:
        import subprocess
        out = subprocess.run([_NVSMI["path"], "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        vals = [int(x) for x in out.split() if x.strip().isdigit()]
        return max(vals) if vals else None
    except Exception:
        return None

def temp_guard(cfg, name):
    """If the GPU is too hot, stop feeding it work until it cools down. Returns seconds paused."""
    t = gpu_temp(); hi, lo = cfg.get("gpu_temp_high", 85), cfg.get("gpu_temp_resume", 75)
    if t is None or t < hi:
        return 0
    log(f"[{name}] GPU {t} C >= {hi} C: pausing until <= {lo} C")
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    while time.time() - t0 < 1800:
        time.sleep(15)
        t = gpu_temp()
        if t is None or t <= lo:
            break
    log(f"[{name}] resumed after {time.time() - t0:.0f} s (GPU {t} C)")
    return time.time() - t0

def keep_awake():
    """Windows: stop the PC from going to sleep while this process runs (screen may still turn off / lock)."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)   # CONTINUOUS | SYSTEM | DISPLAY: no sleep, screen stays on
            return True
        except Exception:
            return False
    return False

def sys_stats():
    """RAM %, free RAM (GB), CPU %, GPU memory reserved by this process (GB)."""
    try:
        import psutil
        vm = psutil.virtual_memory(); ram, free, cpu = vm.percent, vm.available / 1e9, psutil.cpu_percent(interval=None)
    except Exception:
        ram, free, cpu = float("nan"), float("nan"), float("nan")
    gpu = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
    return dict(ram_pct=round(ram, 1), ram_free_gb=round(free, 2), cpu_pct=round(cpu, 1), gpu_mem_gb=round(gpu, 2))

def ram_guard(cfg, name):
    """If system RAM is nearly full, free caches and wait (up to 60 s) instead of letting Windows swap or crash."""
    st = sys_stats()
    if not (st["ram_pct"] >= cfg.get("ram_critical", 95)):
        return False
    import gc
    log(f"[{name}] RAM {st['ram_pct']}% (free {st['ram_free_gb']} GB): pausing to let memory recover")
    for _ in range(12):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(5)
        if sys_stats()["ram_pct"] < cfg.get("ram_critical", 95) - 3:
            break
    return True

# --------------------------------------------------------------------------------------- data preparation
def find_dataset_root(base, need):
    """First folder under `base` whose sub-folder names cover every alternative group in `need`."""
    base = Path(base)
    if not base.exists():
        return None
    for root, dirs, _ in os.walk(base):
        names = {d.lower() for d in dirs}
        if all(any(a in names for a in alts) for alts in need):
            return Path(root)
    return None


def index_pairs(root):
    """Pairs every image with its mask by base name (never by file name with extension)."""
    root = Path(root)
    img_dir = find_dataset_root(root, [("images", "image", "imgs")])
    img_dir = next(p for p in img_dir.iterdir() if p.is_dir() and p.name.lower() in ("images", "image", "imgs"))
    msk_dir = next(p for p in img_dir.parent.iterdir()
                   if p.is_dir() and p.name.lower() in ("annotations", "annotation", "masks", "mask", "labels"))
    masks = {}
    for p in msk_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            sp = p.parent.name.lower() if p.parent != msk_dir else "all"
            masks[(sp, p.stem)] = p
            masks.setdefault(("any", p.stem), p)
    rows, missing = [], []
    for p in sorted(img_dir.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        sp = p.parent.name.lower() if p.parent != img_dir else "all"
        m = masks.get((sp, p.stem)) or masks.get(("any", p.stem))
        if m is None:
            missing.append(str(p)); continue
        rows.append(dict(image=str(p), mask=str(m), official_split=sp))
    df = pd.DataFrame(rows)
    df["uid"] = np.arange(len(df))
    return df, missing, img_dir, msk_dir


def leaf_proxy(img):
    """Rough leaf mask from excess-green + Otsu (PlantSeg has no leaf masks; describe it as a proxy)."""
    f = img.astype(np.float32) / 255.0
    exg = 2 * f[..., 1] - f[..., 0] - f[..., 2]
    exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, leaf = cv2.threshold(exg, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return leaf.astype(np.float32)


def _cache_one(args):
    """Reads one original pair, writes the cached copy (long side = L), returns mask statistics."""
    image, mask, cimg, cmsk, L, size = args
    try:
        need = not (os.path.exists(cimg) and os.path.exists(cmsk))
        img = None if need else cv2.imread(cimg)
        m = None if need else cv2.imread(cmsk, cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:                      # missing or damaged cache (e.g. power cut): rebuild
            need = True
            img = cv2.imread(image); m = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            return None
        if m.shape != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if need:
            s = L / max(img.shape[:2])
            if s < 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            ti, tm = cimg[:-4] + ".tmp.jpg", cmsk[:-4] + ".tmp.png"
            cv2.imwrite(ti, img, [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(tm, m)
            os.replace(ti, cimg); os.replace(tm, cmsk)
        vals, cnt = np.unique(m, return_counts=True)
        fg = vals != 0
        disease = int(vals[fg][np.argmax(cnt[fg])]) if fg.any() else 0
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA)
        leaf = leaf_proxy(small)
        n_leaf = cv2.connectedComponentsWithStats(leaf.astype(np.uint8), connectivity=8)[2][1:, 4]
        n_leaves = int((n_leaf > 0.01 * size * size).sum())
        bm = (m > 0).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(bm, connectivity=8)
        comp = (st[1:, 4] / bm.size).astype(np.float32) if n > 1 else np.zeros(0, np.float32)
        return dict(disease=disease, lesion_frac=float(bm.mean()), bg_share=float(1 - leaf.mean()),
                    n_leaves=n_leaves, n_lesions=int(n - 1), comp_fracs=comp.tolist())
    except Exception:
        return None


def _phash(path):
    import imagehash
    from PIL import Image
    return int(str(imagehash.phash(Image.open(path).convert("RGB"))), 16)


def group_near_duplicates(hashes, thr):
    """Union-find over pairs with pHash Hamming distance <= thr. Returns a group id per image."""
    H = np.array(hashes, dtype=np.uint64)
    n = len(H)
    lut = np.array([bin(i).count("1") for i in range(256)], np.uint8)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for s in range(0, n, 256):
        x = H[s:s + 256, None] ^ H[None, :]
        d = lut[x.view(np.uint8).reshape(x.shape[0], n, 8)].sum(-1)
        ii, jj = np.nonzero(d <= thr)
        for i, j in zip(ii + s, jj):
            if j > i:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def stratified_group_split(df, ratios, seed=0):
    """70/15/15 by disease; whole duplicate groups move together; split by image share within each disease."""
    rng = np.random.RandomState(seed)
    gdis = df.groupby("group").disease.agg(lambda s: s.iloc[0])
    gsize = df.groupby("group").size()
    assign = {}
    for d in sorted(gdis.unique()):
        gids = gdis.index[gdis == d].to_numpy().copy(); rng.shuffle(gids)
        tot = gsize[gids].sum(); acc = 0
        for g in gids:
            pos = (acc + gsize[g] / 2) / tot          # position of the group's midpoint in this disease
            assign[g] = "train" if pos < ratios[0] else ("val" if pos < ratios[0] + ratios[1] else "test")
            acc += gsize[g]
    return df.group.map(assign).to_numpy()


def prepare_data(cfg, n_proc=4):
    """Index, cache, stats and the frozen leak-free split. Safe to call every session (resumes)."""
    from multiprocessing import Pool
    from tqdm.auto import tqdm
    split_csv = wp(cfg, "splits", "split.csv"); split_csv.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(cfg["paths"]["cache"]); (cache / "img").mkdir(parents=True, exist_ok=True); (cache / "msk").mkdir(exist_ok=True)
    fresh = not split_csv.exists()
    if fresh:
        df, missing, img_dir, msk_dir = index_pairs(cfg["paths"]["plantseg_root"])
        log(f"images dir {img_dir} | masks dir {msk_dir} | {len(df)} pairs, {len(missing)} images without a mask")
    else:
        df = pd.read_csv(split_csv)
        log(f"Loaded frozen split {split_csv}: {df.split.value_counts().to_dict()}")
    df["cimg"] = [str(cache / "img" / f"{u:05d}.jpg") for u in df.uid]
    df["cmsk"] = [str(cache / "msk" / f"{u:05d}.png") for u in df.uid]
    args = [(r.image, r.mask, r.cimg, r.cmsk, cfg["cache_long_side"], cfg["img_size"]) for r in df.itertuples()]
    with Pool(n_proc) as pool:
        info = list(tqdm(pool.imap(_cache_one, args, chunksize=16), total=len(args), desc="caching + mask stats"))
    bad = np.array([i is None for i in info])
    if bad.any():
        log(f"WARNING {bad.sum()} unreadable pairs dropped, e.g. {df.image[bad].head(3).tolist()}")
    if fresh:
        keep = ~bad
        df = df[keep].reset_index(drop=True); info = [i for i in info if i is not None]
        for k in ["disease", "lesion_frac", "bg_share", "n_leaves", "n_lesions"]:
            df[k] = [i[k] for i in info]
        comp = np.concatenate([np.array(i["comp_fracs"], np.float32) for i in info])
        q = np.quantile(comp, [0.1, 0.25, 0.5, 0.75, 0.9]) if len(comp) else []
        pd.DataFrame(dict(q=[0.1, 0.25, 0.5, 0.75, 0.9], comp_frac=q)).to_csv(wp(cfg, "splits", "lesion_size_quantiles.csv"), index=False)
        log("lesion component area / image area quantiles (10,25,50,75,90%): " + ", ".join(f"{v:.4%}" for v in q))
        log(f"frozen size bins: small < {cfg['size_bins'][0]:.1%} <= medium <= {cfg['size_bins'][1]:.1%} < large "
            f"-> shares {np.mean(comp < cfg['size_bins'][0]):.1%} / {np.mean((comp >= cfg['size_bins'][0]) & (comp <= cfg['size_bins'][1])):.1%} / {np.mean(comp > cfg['size_bins'][1]):.1%}")
        with Pool(n_proc) as pool:
            hashes = list(tqdm(pool.imap(_phash, df.cimg, chunksize=32), total=len(df), desc="pHash"))
        df["phash"] = [f"{h:016x}" for h in hashes]
        df["group"] = group_near_duplicates(hashes, cfg["phash_thr"])
        gs = df.group.value_counts()
        log(f"near-duplicate groups: {int((gs > 1).sum())} groups hold {int(gs[gs > 1].sum())} images (Hamming <= {cfg['phash_thr']})")
        df["split"] = stratified_group_split(df, cfg["split"], seed=0)
        if cfg.get("use_official_split"):          # dataset ships its own split in images/{train,val,test} (e.g. NLB tiles split by photo)
            off = df.official_split.str.lower().replace({"valid": "val", "validation": "val"})
            if not off.isin(["train", "val", "test"]).all():
                raise ValueError(f"use_official_split: folders must be train/val/test, found {sorted(off.unique())}")
            df["split"] = off.to_numpy()
        atomic_to_csv(df.drop(columns=["cimg", "cmsk"]), split_csv, index=False)
        log(f"Frozen split saved: {df.split.value_counts().to_dict()} | diseases per split "
            f"{df.groupby('split').disease.nunique().to_dict()}")
    else:
        df = df[~bad].reset_index(drop=True)
    return df


def get_frames(cfg, df):
    fr = {s: df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test"]}
    if cfg["quick"]:
        n = cfg["quick_n"]
        fr = {s: f.sample(min(k, len(f)), random_state=0).reset_index(drop=True) for (s, f), k in zip(fr.items(), n)}
    for f in fr.values():
        f["key"] = [Path(p).stem for p in f.cimg]
    return fr


def load_split_frames(cfg):
    df = pd.read_csv(wp(cfg, "splits", "split.csv"))
    cache = Path(cfg["paths"]["cache"])
    df["cimg"] = [str(cache / "img" / f"{u:05d}.jpg") for u in df.uid]
    df["cmsk"] = [str(cache / "msk" / f"{u:05d}.png") for u in df.uid]
    df = df[[os.path.exists(p) for p in df.cimg]].reset_index(drop=True)
    return get_frames(cfg, df)

# --------------------------------------------------------------------------------------- dataset
def size_weight_map(mask):
    """Small lesions weigh more: 1/sqrt(area/median area), clipped to [0, 3]; 0 outside lesions."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    w = np.zeros(mask.shape, np.float32)
    if n <= 1:
        return w
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    lut = np.zeros(n, np.float32)
    lut[1:] = np.clip(1.0 / np.sqrt(areas / np.median(areas)), 0, 3)
    return lut[lab]


def make_aug(size, train):
    import albumentations as A
    if not train:
        return A.Compose([A.Resize(size, size)])
    v2 = int(A.__version__.split(".")[0]) >= 2
    pad = (A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0) if v2 else
           A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0))
    scales = [int(size * s) for s in np.arange(0.5, 2.01, 0.125)]   # random scale 0.5-2.0 (methodology 4.3)
    return A.Compose([A.LongestMaxSize(max_size=scales), pad, A.RandomCrop(size, size),
                      A.HorizontalFlip(), A.VerticalFlip(p=0.2), A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5)])


def to_tensor(img):
    return torch.from_numpy(((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy())


def read_pair(r):
    img = cv2.cvtColor(cv2.imread(r["cimg"] if "cimg" in r else r["image"]), cv2.COLOR_BGR2RGB)
    m = cv2.imread(r["cmsk"] if "cmsk" in r else r["mask"], cv2.IMREAD_GRAYSCALE)
    if m.shape != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img, (m > 0).astype(np.uint8)


class LesionSeg(torch.utils.data.Dataset):
    def __init__(self, frame, size, train, clutter=None):
        self.f = frame.reset_index(drop=True); self.aug = make_aug(size, train)
        self.clutter, self.train = clutter, train

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        img, msk = read_pair(self.f.iloc[i])
        o = self.aug(image=img, mask=msk); img, msk = o["image"], o["mask"]
        out = dict(img=to_tensor(img), mask=torch.from_numpy(msk.astype(np.int64)),
                   leaf=torch.from_numpy(leaf_proxy(img)), idx=i)
        if self.train:
            out["size_w"] = torch.from_numpy(size_weight_map(msk))
        if self.clutter is not None:
            c_img, c_msk, occ = self.clutter(img, msk)
            out.update(c_img=to_tensor(c_img), c_mask=torch.from_numpy(c_msk.astype(np.int64)), occ=torch.from_numpy(occ))
        return out


def _worker_init(_):
    info = torch.utils.data.get_worker_info()
    if info is not None and getattr(info.dataset, "clutter", None) is not None:
        info.dataset.clutter.rng.seed(torch.initial_seed() % 2 ** 32)    # different clutter per worker and epoch


def make_loader(frame, cfg, train, bs, clutter=None, shuffle=None, workers=None):
    ds = LesionSeg(frame, cfg["img_size"], train, clutter)
    w = cfg["num_workers"] if workers is None else workers
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=train if shuffle is None else shuffle,
                                       num_workers=w, drop_last=train and len(ds) > bs, worker_init_fn=_worker_init,
                                       pin_memory=torch.cuda.is_available(), persistent_workers=w > 0,
                                       prefetch_factor=4 if w > 0 else None)   # speed: workers stay alive between validations

# --------------------------------------------------------------------------------------- clutter generator
class Clutter:
    """Leaf-like occluders (elliptical patches cut from other training images), shadows, dirt, glare,
    background swap, blur and JPEG artefacts. Returns cluttered image, lesion mask with occluded lesion
    removed, and the occluder mask (1 = pasted pixel, excluded from the CPD KL term).
    `last_altered` holds the share of altered pixels (clutter level per image)."""
    LEVELS = {"light": dict(n_occ=(1, 1), shadow=0.0, dirt=(0, 10), glare=0.0, bg=0.0),
              "medium": dict(n_occ=(1, 3), shadow=0.6, dirt=(0, 30), glare=0.3, bg=0.3),
              "heavy": dict(n_occ=(3, 3), shadow=1.0, dirt=(30, 60), glare=0.5, bg=0.6)}

    def __init__(self, bank, p=0.5, level="medium", max_cover=0.3, seed=None):
        self.bank, self.p, self.cfg, self.max_cover = bank, p, self.LEVELS[level], max_cover
        self.rng = random.Random(seed); self.last_altered = 0.0

    def _ellipse(self, h, w):
        m = np.zeros((h, w), np.uint8); r = self.rng
        cv2.ellipse(m, (r.randint(0, w - 1), r.randint(0, h - 1)), (r.randint(w // 10, w // 4), r.randint(h // 12, h // 5)),
                    r.uniform(0, 180), 0, 360, 1, -1)
        return cv2.GaussianBlur(m.astype(np.float32), (7, 7), 0)

    def _src(self, h, w):
        s = cv2.resize(self.rng.choice(self.bank), (w, h))
        return s[:, ::-1].copy() if self.rng.random() < 0.5 else s

    def __call__(self, img, msk):
        r, c = self.rng, self.cfg
        img, msk = img.copy(), msk.copy(); h, w = msk.shape
        occ = np.zeros((h, w), np.float32); alt = np.zeros((h, w), np.float32)
        self.last_altered = 0.0
        if r.random() > self.p:
            return img, msk, occ
        lesion_px = max(int(msk.sum()), 1)
        if r.random() < c["bg"]:                           # background swap on non-leaf, non-lesion pixels
            leaf = cv2.dilate(leaf_proxy(img), np.ones((7, 7), np.uint8))
            bgm = cv2.GaussianBlur(((leaf == 0) & (msk == 0)).astype(np.float32), (9, 9), 0)
            if 0.02 < bgm.mean() < 0.85:
                img = (img * (1 - bgm[..., None]) + self._src(h, w) * bgm[..., None]).astype(np.uint8)
                alt = np.maximum(alt, (bgm > 0.5).astype(np.float32))
        for _ in range(r.randint(*c["n_occ"])):           # leaf-like occluders, never > 30% of lesion pixels
            src, a = self._src(h, w), self._ellipse(h, w)
            if msk.sum() > 0 and ((a > 0.5) & (msk > 0)).sum() / lesion_px > self.max_cover:
                continue
            img = (img * (1 - a[..., None]) + src * a[..., None]).astype(np.uint8)
            occ = np.maximum(occ, (a > 0.5).astype(np.float32))
        msk[occ > 0] = 0                                    # label follows the occluder
        alt = np.maximum(alt, occ)
        if r.random() < c["shadow"]:
            pts = np.array([[r.randint(0, w), r.randint(0, h)] for _ in range(5)], np.int32)
            s = np.zeros((h, w), np.float32); cv2.fillPoly(s, [pts], 1.0)
            s = cv2.GaussianBlur(s, (31, 31), 0) * r.uniform(0.2, 0.5)
            img = (img * (1 - s[..., None])).astype(np.uint8)
            alt = np.maximum(alt, (s > 0.05).astype(np.float32))
        for _ in range(r.randint(*c["dirt"])):
            g = r.randint(60, 110); ctr = (r.randint(0, w - 1), r.randint(0, h - 1)); rad = r.randint(1, 4)
            cv2.circle(img, ctr, rad, (g, g, g), -1); cv2.circle(alt, ctr, rad, 1.0, -1)
        if r.random() < c["glare"]:
            g = np.zeros((h, w), np.float32)
            cv2.circle(g, (r.randint(0, w - 1), r.randint(0, h - 1)), r.randint(10, 40), 1.0, -1)
            g = cv2.GaussianBlur(g, (41, 41), 0) * 0.6
            img = np.clip(img.astype(np.float32) + g[..., None] * 255, 0, 255).astype(np.uint8)
            alt = np.maximum(alt, (g > 0.05).astype(np.float32))
        if r.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        if r.random() < 0.5:
            ok, enc = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, r.randint(40, 90)])
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].copy()
        self.last_altered = float(alt.mean())
        return img, msk, occ


def make_bank(frame, cfg, n=300, seed=0):
    """Occluder/background source images: TRAINING images only (no test pixels leak into training)."""
    paths = frame.sample(min(n, len(frame)), random_state=seed).cimg
    s = cfg["img_size"]
    return [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (s, s)) for p in paths]


def build_clutter_test(cfg, test, bank):
    """Frozen synthetic clutter test set: three levels, fixed seeds, built once and reused."""
    from tqdm.auto import tqdm
    s = cfg["img_size"]; out = {}
    for level, sd in [("light", 11), ("medium", 22), ("heavy", 33)]:
        d = wp(cfg, "clutter_test", level); csv = d / "index.csv"
        if csv.exists():
            out[level] = pd.read_csv(csv, dtype={"key": str}); continue
        (d / "images").mkdir(parents=True, exist_ok=True); (d / "masks").mkdir(exist_ok=True); (d / "occluders").mkdir(exist_ok=True)
        cl = Clutter(bank, p=1.0, level=level, seed=sd); rows = []
        for _, r in tqdm(test.iterrows(), total=len(test), desc=f"clutter test {level}"):
            img, msk = read_pair(r)
            img = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA); msk = cv2.resize(msk, (s, s), interpolation=cv2.INTER_NEAREST)
            ci, cm, occ = cl(img, msk)
            ip, mp, op = d / "images" / f"{r.key}.png", d / "masks" / f"{r.key}.png", d / "occluders" / f"{r.key}.png"
            cv2.imwrite(str(ip), cv2.cvtColor(ci, cv2.COLOR_RGB2BGR)); cv2.imwrite(str(mp), cm * 255); cv2.imwrite(str(op), (occ * 255).astype(np.uint8))
            rows.append(dict(image=str(ip), mask=str(mp), key=r.key, disease=r.disease, altered=cl.last_altered,
                             bg_share=r.bg_share, lesion_frac=r.lesion_frac))
        out[level] = pd.DataFrame(rows); out[level].to_csv(csv, index=False)
    return out

# --------------------------------------------------------------------------------------- models
def build(arch, encoder, classes=2, pretrained=True):
    import segmentation_models_pytorch as smp
    return getattr(smp, arch)(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, classes=classes)


def forward_with_feats(model, x):
    """Logits and the encoder pyramid [input, s2, s4, s8, s16, s32] (indices 3, 4, 5 = strides 8, 16, 32)."""
    feats = model.encoder(x)
    try:
        dec = model.decoder(feats)          # newer smp
    except TypeError:
        dec = model.decoder(*feats)         # older smp
    return model.segmentation_head(dec), feats


class Adapters(nn.Module):
    """1x1 convs from student to teacher channels; training only, dropped at export."""
    def __init__(self, s_ch, t_ch, stages):
        super().__init__(); self.stages = list(stages)
        self.convs = nn.ModuleList([nn.Conv2d(s_ch[i], t_ch[i], 1) for i in self.stages])

    def forward(self, s_feats):
        return [conv(s_feats[i]) for conv, i in zip(self.convs, self.stages)]


def n_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6

# --------------------------------------------------------------------------------------- losses
_DICE = None


def seg_loss(logits, y):
    global _DICE
    if _DICE is None:
        import segmentation_models_pytorch as smp
        _DICE = smp.losses.DiceLoss(mode="multiclass")
    return F.cross_entropy(logits, y) + _DICE(logits, y)


def pixel_kl(t, s, tau):
    """Per-pixel KL(teacher || student) on softened logits, x tau^2. Shape (B, H, W)."""
    return (F.softmax(t / tau, 1) * (F.log_softmax(t / tau, 1) - F.log_softmax(s / tau, 1))).sum(1) * tau ** 2


def logit_kd(t, s, tau):
    return pixel_kl(t, s, tau).mean()


def cwd_loss(t, s, tau):
    """Channel-wise distillation (Shu et al.): KL between each channel's spatial softmax."""
    B, C, H, W = t.shape
    pt = F.softmax(t.reshape(B, C, -1) / tau, -1)
    return (pt * (F.log_softmax(t.reshape(B, C, -1) / tau, -1) - F.log_softmax(s.reshape(B, C, -1) / tau, -1))).sum(-1).mean() * tau ** 2


def attention(f):
    return F.normalize(f.pow(2).mean(1).flatten(1), dim=1)     # Zagoruyko & Komodakis


def resize_to(x, ref):
    return x if x.shape[-2:] == ref.shape[-2:] else F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def at_loss(t_feats, s_feats, stages, valid=None):
    loss = 0.0
    for i in stages:
        t_f, s_f = t_feats[i], resize_to(s_feats[i], t_feats[i])
        if valid is not None:
            v = F.interpolate(valid.unsqueeze(1), size=t_f.shape[-2:], mode="nearest")
            t_f, s_f = t_f * v, s_f * v
        loss = loss + (attention(t_f) - attention(s_f)).pow(2).sum(1).mean()
    return loss


def skd_pairwise(t_f, s_f, grid=12):
    """SKD pair-wise term (Liu et al.): match cosine-similarity graphs of pooled feature nodes."""
    t = F.normalize(F.adaptive_avg_pool2d(t_f, grid).flatten(2), dim=1)
    s = F.normalize(F.adaptive_avg_pool2d(s_f, grid).flatten(2), dim=1)
    return (t.transpose(1, 2) @ t - s.transpose(1, 2) @ s).pow(2).mean()


def boundary_band(y, k=5):
    y = y.float().unsqueeze(1)
    band = (F.max_pool2d(y, k, 1, k // 2) + F.max_pool2d(-y, k, 1, k // 2)).clamp(0, 1)   # dilation - erosion
    return F.avg_pool2d(band, 3, 1, 1).squeeze(1)


def bpkd_loss(t, s, y, tau, k=5):
    """BPKD (Liu et al. WACV 2024): KD on edge pixels normalised by edge area + KD on body pixels."""
    e = (boundary_band(y, k) > 0).float(); kl = pixel_kl(t, s, tau)
    return (e * kl).sum() / e.sum().clamp(min=1) + ((1 - e) * kl).sum() / (1 - e).sum().clamp(min=1)


def importance_map(y, size_w, leaf, t_logits, kd):
    """LSD importance W = 1 + a*boundary + b*small-lesion + g*teacher confidence + d*leaf context; mean 1."""
    conf = F.softmax(t_logits, 1).max(1).values
    W = (1 + kd["alpha_boundary"] * boundary_band(y, kd["boundary_k"]) + kd["beta_size"] * size_w
         + kd["gamma_conf"] * conf + kd["delta_leaf"] * leaf * (1 - y.float()))
    return W / W.mean(dim=(1, 2), keepdim=True)


def sard_map(t_logits):
    """SARD-style structure importance (re-implemented from the paper description, no released code used):
    boundary salience of the teacher prediction + local structure density + teacher confidence; mean 1."""
    p = F.softmax(t_logits, 1)[:, 1:2]
    gx = F.pad(p[..., :, 1:] - p[..., :, :-1], (0, 1)); gy = F.pad(p[..., 1:, :] - p[..., :-1, :], (0, 0, 0, 1))
    sal = (gx ** 2 + gy ** 2).sqrt()
    sal = sal / sal.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    dens = F.avg_pool2d(sal, 15, 1, 7); dens = dens / dens.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    conf = F.softmax(t_logits, 1).max(1, keepdim=True).values
    W = (1 + sal + dens + conf).squeeze(1)
    return W / W.mean(dim=(1, 2), keepdim=True)


def chan_norm(f):
    return (f - f.mean(dim=(2, 3), keepdim=True)) / (f.var(dim=(2, 3), keepdim=True, unbiased=False) + 1e-5).sqrt()


def lsd_loss(t_logits, s_logits, t_feats, s_adapted, stages, W, kd):
    """Contribution 1: importance-weighted pixel KL + importance-weighted feature distillation."""
    kl = (W * pixel_kl(t_logits, s_logits, kd["tau"])).sum() / W.sum()
    feat = 0.0
    for s_f, i in zip(s_adapted, stages):
        t_f = t_feats[i]; s_f = resize_to(s_f, t_f)
        Wl = F.interpolate(W.unsqueeze(1), size=t_f.shape[-2:], mode="bilinear", align_corners=False)
        d = (chan_norm(t_f) - chan_norm(s_f)).pow(2).mean(1, keepdim=True)
        feat = feat + (Wl * d).sum() / Wl.sum()
    return kl + kd["mu_feat"] * feat


def cpd_loss(t_logits_clean, s_logits_clutter, t_feats_clean, s_feats_clutter, stages, occ, kd):
    """Contribution 2: clean-view teacher vs cluttered-view student, pasted pixels excluded."""
    valid = 1 - occ
    kl = (valid * pixel_kl(t_logits_clean, s_logits_clutter, kd["tau"])).sum() / valid.sum().clamp(min=1)
    return kd["cpd_kl"] * kl + kd["rho_att"] * at_loss(t_feats_clean, s_feats_clutter, stages, valid)


def loss_unit_tests(cfg):
    kd = cfg["kd"]
    t = torch.randn(2, 2, 32, 32); y = (torch.rand(2, 32, 32) > 0.8).long()
    W = importance_map(y, torch.rand(2, 32, 32), torch.rand(2, 32, 32), t, kd)
    assert abs(W.mean().item() - 1) < 1e-4 and abs(sard_map(t).mean().item() - 1) < 1e-4
    f = [torch.randn(2, 8, 64 // 2 ** i, 64 // 2 ** i) for i in range(6)]
    checks = dict(pixel_kl=pixel_kl(t, t, 4).abs().max(), cwd=cwd_loss(t, t, 4).abs(), bpkd=bpkd_loss(t, t, y, 4).abs(),
                  skd=skd_pairwise(f[5], f[5]).abs(), at=at_loss(f, f, [3, 4, 5]).abs(),
                  lsd=lsd_loss(t, t, f, [f[i] for i in [3, 4, 5]], [3, 4, 5], W, kd).abs(),
                  cpd=cpd_loss(t, t, f, f, [3, 4, 5], torch.zeros(2, 32, 32), kd).abs())
    for k, v in checks.items():
        assert float(v) < 1e-4, (k, float(v))
    s = torch.randn(2, 2, 32, 32)
    for k, v in dict(lsd=lsd_loss(t, s, f, [f[i] * 1.5 + 1 for i in [3, 4, 5]], [3, 4, 5], W, kd),
                     cpd=cpd_loss(t, s, f, [x.flip(-1) for x in f], [3, 4, 5], torch.zeros(2, 32, 32), kd)).items():
        assert v.dim() == 0 and float(v) > 0, k
    return "loss unit tests passed: all terms are scalars, zero when student == teacher, W has mean 1"

# --------------------------------------------------------------------------------------- methods & jobs
METHODS = {
    # name: teacher needed, LSD, CPD, baseline term, adapters, kd overrides
    "A0_student":   dict(teacher=False),
    "B_logitKD":    dict(teacher=True, base="logit"),
    "B_AT":         dict(teacher=True, base="at"),
    "B_CWD":        dict(teacher=True, base="cwd"),
    "B_SKD":        dict(teacher=True, base="skd"),
    "B_BPKD":       dict(teacher=True, base="bpkd"),
    "B_SARD":       dict(teacher=True, base="sard", adapt=True),
    "B_Caug":       dict(teacher=True, base="caug"),     # key control for CPD: teacher sees the SAME cluttered image
    "A1_uniformKD": dict(teacher=True, lsd=True, adapt=True, kd=dict(alpha_boundary=0, beta_size=0, gamma_conf=0, delta_leaf=0)),
    "A2_boundary":  dict(teacher=True, lsd=True, adapt=True, kd=dict(beta_size=0, gamma_conf=0, delta_leaf=0)),
    "A3_size":      dict(teacher=True, lsd=True, adapt=True, kd=dict(gamma_conf=0, delta_leaf=0)),
    "A4_conf":      dict(teacher=True, lsd=True, adapt=True, kd=dict(delta_leaf=0)),
    "A5_LSD":       dict(teacher=True, lsd=True, adapt=True),
    "A6_CPD":       dict(teacher=True, cpd=True),
    "A7_full":      dict(teacher=True, lsd=True, cpd=True, adapt=True),
    # tuning round 2: tuning round 1 showed CWD best on clean images and CPD best under clutter -> combine them
    "X_cwdCPD":     dict(teacher=True, base="cwd", cpd=True),
    "X_cwdCPD_c75": dict(teacher=True, base="cwd", cpd=True, kd=dict(clutter_prob=0.75)),
    "X_cwdLSDCPD":  dict(teacher=True, base="cwd", lsd=True, cpd=True, adapt=True, kd=dict(lambda_lsd=0.5)),
}
SENSITIVITY = {   # all variants of A7_full, MobileNetV3 student, seed 0 (methodology 5.3 "also report")
    "S_tau1": dict(tau=1.0), "S_tau2": dict(tau=2.0),
    "S_lsd0.5": dict(lambda_lsd=0.5), "S_lsd2": dict(lambda_lsd=2.0), "S_lsd5": dict(lambda_lsd=5.0),
    "S_cpd0.5": dict(lambda_cpd=0.5), "S_cpd2": dict(lambda_cpd=2.0), "S_cpd5": dict(lambda_cpd=5.0),
    "S_clut0.25": dict(clutter_prob=0.25), "S_clut0.75": dict(clutter_prob=0.75),
    "S_cpdLogitsOnly": dict(rho_att=0.0), "S_cpdAttOnly": dict(cpd_kl=0.0),
    # tuning pilot (pilot showed the LSD loss ~8x the segmentation loss and CPD ~ plain clutter augmentation)
    "S_feat0.25": dict(mu_feat=0.25), "S_lsd1": dict(lambda_lsd=1.0),
    "S_combo": dict(mu_feat=0.25, lambda_lsd=0.5, rho_att=0.0, clutter_prob=0.75),
}
TUNE_BASE = ["A0_student", "B_AT", "B_CWD", "B_SARD", "B_Caug", "A6_CPD", "A7_full"]
TUNE_VARIANTS = ["S_feat0.25", "S_lsd0.5", "S_cpdLogitsOnly", "S_clut0.75", "S_combo"]
TUNE_VARIANTS2 = ["X_cwdCPD", "X_cwdCPD_c75", "X_cwdLSDCPD"]
for _k, _v in SENSITIVITY.items():
    METHODS[_k] = dict(METHODS["A7_full"], kd=_v)

# Final method set (cfg["method_set"] == "v2"), chosen by the tuning pilot (2 seeds):
#   CWD + LSD(0.5) + CPD matched CWD on clean images (-0.9, within noise) and beat it by +2.8 / +4.0 mIoU under
#   medium / heavy clutter, and C-aug by +1.6 to +1.9 everywhere. Components are ablated on top of CWD.
METHODS_V2 = dict(METHODS)
METHODS_V2.update({
    "A5_LSD":     dict(teacher=True, base="cwd", lsd=True, adapt=True, kd=dict(lambda_lsd=0.5)),   # CWD + LSD
    "A6_CPD":     dict(teacher=True, base="cwd", cpd=True),                                         # CWD + CPD
    "A7_full":    dict(teacher=True, base="cwd", lsd=True, cpd=True, adapt=True, kd=dict(lambda_lsd=0.5)),
    "A8_CPDonly": dict(teacher=True, cpd=True),       # CPD without CWD: like-for-like with the C-aug control
    # REV-1 factorial ablation (reviewer request). All like-for-like with C-aug: no CWD, no LSD, attention off
    # unless stated. Factors: target view (clean / cluttered) x occlusion mask (on / off); A8 adds attention.
    "R1_cleanMask":     dict(teacher=True, cpd=True, kd=dict(rho_att=0.0)),                    # clean target, mask, no att.
    "R5_cleanNoMask":   dict(teacher=True, cpd=True, cpd_nomask=True, kd=dict(rho_att=0.0)),   # clean target, no mask, no att.
    "R3_clutterMask":   dict(teacher=True, base="caug", caug_mask=True),                       # cluttered target, mask
    "R4_augOnly":       dict(teacher=False, clutter_only=True),                                # clutter + seg loss, no KD
    "R6_augCWD":        dict(teacher=True, base="cwd", clutter_only=True),                     # clutter + seg loss, CWD on clean only
    "R7_augCWDcpd":     dict(teacher=True, base="cwd", cpd=True, kd=dict(rho_att=0.0)),        # = R6 + CPD (no attention)
})
for _k, _v in SENSITIVITY.items():
    METHODS_V2[_k] = dict(METHODS_V2["A7_full"], kd={**METHODS_V2["A7_full"]["kd"], **_v})

def method_spec(cfg, method):
    return (METHODS_V2 if cfg.get("method_set") == "v2" else METHODS).get(method, {})

FIRST_KD = ["B_logitKD", "B_AT", "B_CWD", "B_BPKD", "B_SARD", "B_Caug", "A5_LSD", "A6_CPD", "A7_full"]
ABLATIONS = ["A1_uniformKD", "A2_boundary", "A3_size", "A4_conf"]
BASELINES = ["B_logitKD", "B_AT", "B_CWD", "B_BPKD", "B_SARD", "B_Caug"]


# --------------------------------------------------------------------------------------- teacher selection
def teacher_run(cfg, key):
    return f"teacher-{key}__teacher__s0"

def chosen_teacher(cfg):
    """The teacher every distillation run uses: cfg['teacher_choice'] if fixed, else ('auto') the trained
    candidate with the highest validation mIoU. None until a candidate has finished."""
    if cfg["teacher_choice"] != "auto":
        return cfg["teacher_choice"] if run_done(cfg, teacher_run(cfg, cfg["teacher_choice"])) else None
    if any(not run_done(cfg, teacher_run(cfg, k)) for k in cfg["teachers_to_train"]):
        return None                                   # wait until every candidate is trained, then choose
    sc = {k: json.load(open(wp(cfg, "runs", teacher_run(cfg, k), "done.json")))["best_val_miou"] for k in cfg["teachers_to_train"]}
    return max(sc, key=sc.get)

def teacher_spec(cfg):
    k = chosen_teacher(cfg)
    return None if k is None else dict(key=k, run=teacher_run(cfg, k), **cfg["teachers"][k])


def run_done(cfg, name):
    return wp(cfg, "runs", name, "done.json").exists()


def eval_done(cfg, name):
    return wp(cfg, "results", "evaldone", name).exists()


def val_scores(cfg):
    rows = []
    for p in wp(cfg, "runs").glob("*/done.json"):
        rows.append(json.load(open(p)))
    return pd.DataFrame(rows)


def best_baseline(cfg, student_short):
    v = val_scores(cfg)
    if v.empty:
        return None
    v = v[(v.model == student_short) & v.method.isin(BASELINES)]
    if v.empty:
        return None
    return v.groupby("method").best_val_miou.mean().idxmax()


def phases(cfg):
    """Ordered phases of the experiment grid. Each phase: (title, gate checked before it, job list).
    Trimmed grid: primary student gets every method; secondary student only A0, best baseline, A7.
    The 'full' gate is checked after seed 0, so a failed idea stops before the other seeds are spent."""
    S, S2 = cfg["students"], cfg.get("secondary_students", []); seeds = cfg["seeds"]; q = cfg["quick"]
    final_seeds = cfg["final_seeds"][:1] if q else cfg["final_seeds"]
    if q:
        seeds = seeds[:1]                       # sanity / pilot: one seed
    J = lambda st, m, sd: dict(name=f"{short(st)}__{m}__s{sd}", kind="student", encoder=st, arch="Unet", method=m, seed=sd)
    p1 = [dict(name=teacher_run(cfg, k), kind="teacher", encoder=cfg["teachers"][k]["encoder"], arch=cfg["teachers"][k]["arch"],
               method="teacher", seed=0, transformer=cfg["teachers"][k].get("transformer", False)) for k in cfg["teachers_to_train"]]
    p1 += [J(st, "A0_student", sd) for sd in seeds for st in S]
    p1 += [dict(name=f"{d['name']}__direct__s{sd}", kind="direct", encoder=d["encoder"], arch=d["arch"], method="direct",
                seed=sd, transformer=d["transformer"]) for sd in seeds for d in cfg["direct"]]
    sec = lambda sds: [J(st, m, sd) for sd in sds for st in S2 for m in cfg["secondary_methods"]]
    sens = [m for m in cfg["sensitivity"] if m in SENSITIVITY]
    first_kd = cfg.get("first_kd") or FIRST_KD                 # seed 0
    seed_kd = cfg.get("seed_kd") or first_kd                   # seeds 1, 2
    ablations = ABLATIONS if cfg.get("ablations") is None else cfg["ablations"]
    out = [("P1 teacher(s) + student alone + direct small models", None, p1),
           ("P2 baselines + LSD/CPD/full, primary student, seed 0", "teacher", [J(st, m, seeds[0]) for st in S for m in first_kd]),
           ("P3 other seeds + second student (A0, best baseline, A7)", "full",
            [J(st, m, sd) for sd in seeds[1:] for st in S for m in seed_kd] + sec(seeds)),
           ("P4 ablations A1-A4, primary student", None, [J(st, m, sd) for sd in seeds for st in S for m in ablations]),
           ("P5 sensitivity (A7 variants, primary student, seed 0)", None, [J(S[0], m, seeds[0]) for m in (sens[:2] if q else sens)])]
    if cfg.get("tune"):                         # tuning pilot: reuses the pilot's teachers and seed-0 runs
        st = S[0]
        return [out[0],
                ("T1 baselines + A7, second seed", None, [J(st, m, 1) for m in TUNE_BASE]),
                ("T2 tuned A7 variants, seeds 0 and 1", None, [J(st, m, sd) for sd in (0, 1) for m in TUNE_VARIANTS]),
                ("T3 CWD + CPD combinations, seeds 0 and 1", None, [J(st, m, sd) for sd in (0, 1) for m in TUNE_VARIANTS2])]
    fin = [J(st, m, sd) for sd in final_seeds for st in S + S2 for m in ["A0_student", "best_baseline", "A7_full"]]
    if fin:
        out.append(("P6 extra seeds for the final table", None, fin))
    out = out[:cfg.get("max_phase", 99)]
    if cfg.get("max_phase", 99) < 3:            # pilot: still check A7 vs the best baseline after P2
        out.append(("P2 check: A7 vs best baseline", "full", []))
    return out


def resolve(cfg, job):
    if job["method"] == "best_baseline":
        # the best baseline is chosen on the PRIMARY student's validation scores (seed-averaged)
        b = best_baseline(cfg, short(job["encoder"])) or best_baseline(cfg, short(cfg["students"][0])) or "B_CWD"
        job = dict(job, method=b, name=f"{short(job['encoder'])}__{b}__s{job['seed']}")
    return job


def gate(cfg, which):
    """Week-3 gate: teacher beats every student-alone model by >= 5 mIoU (val).
    Week-7 gate: full model beats the best baseline by >= 1.5 mIoU (val, mean over seeds), per student."""
    v = val_scores(cfg); g = cfg["gates"]
    if which == "teacher":
        spec = teacher_spec(cfg)
        t = v[v.name == spec["run"]].best_val_miou.max() if spec else float("nan")
        a0 = v[v.method == "A0_student"].groupby("model").best_val_miou.mean()
        dirm = v[v.method == "direct"].groupby("model").best_val_miou.mean()
        best_small = max(a0.max() if len(a0) else 0, dirm.max() if len(dirm) else 0)
        ok = t - best_small >= g["teacher_margin"]
        return ok, f"teacher {spec['key'] if spec else '?'} val mIoU {t:.4f} vs best small model {best_small:.4f} (margin {t - best_small:+.4f}, need {g['teacher_margin']:+.3f})"
    msgs, oks = [], []
    for st in cfg["students"]:
        s = short(st); vv = v[v.model == s]
        full = vv[vv.method == "A7_full"].best_val_miou.mean()
        base = vv[vv.method.isin(BASELINES)].groupby("method").best_val_miou.mean()
        bb = base.idxmax(); m = full - base.max()
        oks.append(m >= g["full_margin"])
        msgs.append(f"{s}: A7_full {full:.4f} vs best baseline {bb} {base.max():.4f} (margin {m:+.4f}, need {g['full_margin']:+.3f})")
    return all(oks), " | ".join(msgs)


def claim(cfg, name, gpu):
    d = wp(cfg, "claims"); d.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(d / f"{name}.claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()} gpu={gpu} t={time.time()}".encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


def release(cfg, name):
    try:
        os.remove(wp(cfg, "claims", f"{name}.claim"))
    except FileNotFoundError:
        pass


def next_job(cfg, gpu):
    """Returns ('job', job) | ('wait', msg) | ('finished', msg) | ('stop', msg)."""
    for title, g, jobs in phases(cfg):
        if g is not None:
            stop_file = wp(cfg, f"GATE_{g}.txt")
            if not stop_file.exists():
                ok, msg = gate(cfg, g)
                atomic_write_text(stop_file, ("PASS " if ok else "FAIL ") + msg + "\n")
                log(f"GATE {g}: {'PASS' if ok else 'FAIL'} - {msg}")
            txt = stop_file.read_text()
            if txt.startswith("FAIL") and cfg["gates"]["stop_on_fail"]:
                return "stop", f"gate '{g}' failed: {txt.strip()}. Methodology 6.2: fix the teacher/idea before continuing. " \
                               f"To continue anyway set CFG['gates']['stop_on_fail']=False and delete {stop_file.name}."
        jobs = [resolve(cfg, j) for j in jobs]
        pending = [j for j in jobs if not (run_done(cfg, j["name"]) and eval_done(cfg, j["name"]))]
        if not pending:
            continue
        for j in pending:
            if j["kind"] == "student" and method_spec(cfg, j["method"]).get("teacher") and teacher_spec(cfg) is None:
                continue          # cannot start before the teacher exists
            if claim(cfg, j["name"], gpu):
                return "job", j
        if any(not run_done(cfg, j["name"]) for j in jobs):
            return "wait", f"{title}: remaining jobs are running on another GPU"
        return "wait", f"{title}: evaluation still running elsewhere"
    return "finished", "all phases complete"

# --------------------------------------------------------------------------------------- training
_TEACHER = {}


def load_teacher(cfg, dev):
    if "m" not in _TEACHER:
        spec = teacher_spec(cfg)
        t = build(spec["arch"], spec["encoder"], pretrained=False).to(dev)
        t.load_state_dict(torch.load(wp(cfg, "runs", spec["run"], "best.pt"), map_location=dev)); t.eval()
        for p in t.parameters():
            p.requires_grad_(False)
        _TEACHER["m"] = t
    return _TEACHER["m"]


@torch.no_grad()
def val_miou(model, dl, dev, amp):
    model.eval(); inter = torch.zeros(2, device=dev); union = torch.zeros(2, device=dev)
    for b in dl:
        with torch.autocast("cuda", enabled=amp):
            p = model(b["img"].to(dev, non_blocking=True)).argmax(1)
        y = b["mask"].to(dev)
        for c in (0, 1):
            inter[c] += ((p == c) & (y == c)).sum(); union[c] += ((p == c) | (y == c)).sum()
    iou = (inter / union.clamp(min=1)).cpu().numpy()
    return dict(miou=float(iou.mean()), iou_lesion=float(iou[1]))


def train_run(cfg, job, frames, bank, dev, deadline):
    name = job["name"]; rd = wp(cfg, "runs", name); rd.mkdir(parents=True, exist_ok=True)
    if (rd / "done.json").exists():
        return "done"
    kind, method, seed = job["kind"], job["method"], job["seed"]
    M = method_spec(cfg, method) if kind == "student" else {}
    kd = {**cfg["kd"], **M.get("kd", {})}; stages = kd["feat_stages"]
    seed_all(seed)
    amp = dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] >= 7
    if kind == "teacher":
        epochs, bs = cfg["teacher_epochs"], cfg["teacher_bs"]
        lr = cfg["lr_transformer"] if job.get("transformer") else cfg["lr_teacher"]
    else:
        epochs, bs = cfg["student_epochs"], cfg["student_bs"]
        lr = cfg["lr_transformer"] if job.get("transformer") else cfg["lr_student"]
    if cfg["quick"]:
        epochs = cfg["quick_epochs"]
    teacher = load_teacher(cfg, dev) if M.get("teacher") else None
    need_clutter = bool(M.get("cpd")) or M.get("base") == "caug" or bool(M.get("clutter_only"))
    clutter = Clutter(bank, p=kd["clutter_prob"], level="medium", seed=seed) if need_clutter else None
    workers = cfg["num_workers"]
    tr = make_loader(frames["train"], cfg, True, bs, clutter=clutter, workers=workers)
    va = make_loader(frames["val"], cfg, False, cfg["eval_bs"])
    model = build(job["arch"], job["encoder"], pretrained=cfg["pretrained"]).to(dev)
    adapt = Adapters(model.encoder.out_channels, teacher.encoder.out_channels, stages).to(dev) if M.get("adapt") else None
    params = list(model.parameters()) + (list(adapt.parameters()) if adapt else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg["weight_decay"])
    total = epochs * len(tr); warm = max(1, min(cfg["warmup_iters"], total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / warm) * max(0.0, 1 - it / total) ** 0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, hist, bad = 0, -1.0, [], 0
    ck = rd / "last.pt"
    if ck.exists():                                                 # resume after a disconnect / new session
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"])
        if adapt is not None and s.get("adapt"):
            adapt.load_state_dict(s["adapt"])
        start, best, hist, bad = s["epoch"] + 1, s["best"], s["log"], s.get("bad", 0)
        log(f"[{name}] resumed at epoch {start}")
    t_run = time.time()
    for ep in range(start, epochs):
        est = np.mean([h["sec"] for h in hist[-3:]]) if hist else 0
        if deadline and time.time() + 1.15 * est + 600 > deadline:
            log(f"[{name}] paused before epoch {ep} (session time budget); resumes next session")
            return "paused"
        st = sys_stats()
        if st["ram_pct"] >= cfg.get("ram_high", 88) and workers > 0:     # slow down: fewer data-loading processes
            workers = max(0, workers // 2); del tr
            import gc; gc.collect()
            tr = make_loader(frames["train"], cfg, True, bs, clutter=clutter, workers=workers)
            log(f"[{name}] RAM {st['ram_pct']}% (free {st['ram_free_gb']} GB): data workers reduced to {workers}")
        cool_s = temp_guard(cfg, name)
        model.train(); t0 = time.time(); parts = collections.defaultdict(lambda: torch.zeros((), device=dev)); nb = 0; paused = 0
        if adapt is not None:
            adapt.train()
        for b in tr:
            x, y = b["img"].to(dev, non_blocking=True), b["mask"].to(dev, non_blocking=True)
            with torch.autocast("cuda", enabled=amp):
                if teacher is not None:
                    with torch.no_grad():
                        t_out, t_feats = forward_with_feats(teacher, x)
                s_out, s_feats = forward_with_feats(model, x)
            s_out = s_out.float(); loss = seg_loss(s_out, y); parts["seg"] += loss.detach()
            if teacher is not None:
                t_out = t_out.float(); t_feats = [f.float() for f in t_feats]; s_feats = [f.float() for f in s_feats]
                base, l = M.get("base"), None
                if base == "logit":
                    l = logit_kd(t_out, s_out, kd["tau"])
                elif base == "at":
                    l = at_loss(t_feats, s_feats, stages)
                elif base == "cwd":
                    l = cwd_loss(t_out, s_out, kd["tau"])
                elif base == "skd":
                    l = logit_kd(t_out, s_out, kd["tau"]) + kd["skd_w"] * skd_pairwise(t_feats[5], s_feats[5])
                elif base == "bpkd":
                    l = bpkd_loss(t_out, s_out, y, kd["tau"], kd["boundary_k"])
                elif base == "sard":
                    l = lsd_loss(t_out, s_out, t_feats, adapt(s_feats), stages, sard_map(t_out), kd)
                if l is not None:
                    loss = loss + l; parts[base] += l.detach()
                if M.get("lsd"):
                    W = importance_map(y, b["size_w"].to(dev), b["leaf"].to(dev), t_out, kd)
                    l = lsd_loss(t_out, s_out, t_feats, adapt(s_feats), stages, W, kd)
                    loss = loss + kd["lambda_lsd"] * l; parts["lsd"] += l.detach()
            if need_clutter:
                xc, yc, occ = b["c_img"].to(dev), b["c_mask"].to(dev), b["occ"].to(dev)
                with torch.autocast("cuda", enabled=amp):
                    sc_out, sc_feats = forward_with_feats(model, xc)
                    if M.get("base") == "caug":
                        with torch.no_grad():
                            tc_out, _ = forward_with_feats(teacher, xc)
                sc_out = sc_out.float(); sc_feats = [f.float() for f in sc_feats]
                lc = seg_loss(sc_out, yc); loss = loss + lc; parts["seg_clutter"] += lc.detach()
                if not M.get("clutter_only"):
                    if M.get("base") == "caug" and M.get("caug_mask"):        # cluttered-view target, occluded pixels excluded
                        valid = 1 - occ
                        l = (valid * pixel_kl(tc_out.float(), sc_out, kd["tau"])).sum() / valid.sum().clamp(min=1)
                    elif M.get("base") == "caug":
                        l = logit_kd(tc_out.float(), sc_out, kd["tau"])
                    else:
                        l = cpd_loss(t_out, sc_out, t_feats, sc_feats, stages, torch.zeros_like(occ) if M.get("cpd_nomask") else occ, kd)
                    loss = loss + kd["lambda_cpd"] * l; parts["clutter_kd"] += l.detach()
            if nb % 25 == 0 and not torch.isfinite(loss):    # checked every 25 batches: each check makes the CPU wait for the GPU
                raise FloatingPointError(f"[{name}] non-finite loss at epoch {ep}: { {k: float(v) for k, v in parts.items()} }")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); nb += 1
            if nb % 50 == 0 and ram_guard(cfg, name):
                paused += 1
            if nb % 25 == 0:
                cool_s += temp_guard(cfg, name)
        do_val = (ep + 1) % cfg.get("val_every", 1) == 0 or ep == epochs - 1
        v = val_miou(model, va, dev, amp) if do_val else dict(miou=None, iou_lesion=None)
        rec = dict(epoch=ep, val_miou=v["miou"], val_iou_lesion=v["iou_lesion"], sec=round(time.time() - t0, 1),
                   lr=opt.param_groups[0]["lr"], workers=workers, ram_pauses=paused, cooldown_s=round(cool_s),
                   gpu_temp=gpu_temp(), **sys_stats(),
                   **{k: round(float(x) / max(nb, 1), 5) for k, x in parts.items()})
        hist.append(rec); log(f"[{name}] {rec}")
        if do_val:
            if v["miou"] > best:
                best, bad = v["miou"], 0; torch.save(model.state_dict(), rd / "best.pt.tmp"); os.replace(rd / "best.pt.tmp", rd / "best.pt")
            else:
                bad += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                        epoch=ep, best=best, bad=bad, log=hist, adapt=adapt.state_dict() if adapt is not None else None), str(ck) + ".tmp")
        os.replace(str(ck) + ".tmp", ck)                # atomic: a power cut mid-save never leaves a broken checkpoint
        pd.DataFrame(hist).to_csv(rd / "log.csv", index=False)
        if cfg.get("patience_evals") and bad >= cfg["patience_evals"]:
            log(f"[{name}] early stop at epoch {ep}: no val improvement in {bad} checks"); break
    model_name, _, _ = parse_run(name)
    res = dict(name=name, model=model_name, kind=kind, arch=job["arch"], encoder=job["encoder"], method=method, seed=seed,
               best_val_miou=best, params_M=round(n_params(model), 3), epochs=epochs, epochs_run=len(hist),
               train_hours=round(sum(h["sec"] for h in hist) / 3600, 3), quick=cfg["quick"], kd=kd if teacher is not None else None)
    atomic_write_text(rd / "done.json", json.dumps(res, indent=1))
    ck.unlink(missing_ok=True)
    return "done"

# --------------------------------------------------------------------------------------- evaluation
def lesion_match(p, y, bins, thr=0.5):
    """Lesion-level TP/FP/FN (a GT lesion is found if one predicted component has IoU >= thr with it),
    plus TP/FN per lesion size bin (component area / image area)."""
    ny, ly = cv2.connectedComponents(y); npr, lp = cv2.connectedComponents(p)
    out = dict(tp=0, fp=0, fn=0, tp_small=0, fn_small=0, tp_medium=0, fn_medium=0, tp_large=0, fn_large=0)
    if ny <= 1:
        out["fp"] = npr - 1; return out
    pa = np.bincount(lp.ravel(), minlength=npr); ga = np.bincount(ly.ravel(), minlength=ny)
    pair = np.bincount((ly.astype(np.int64) * npr + lp).ravel(), minlength=ny * npr).reshape(ny, npr)
    matched = set()
    for g in range(1, ny):
        inter = pair[g, 1:]; iou = inter / (ga[g] + pa[1:] - inter + 1e-9)
        hit = iou.size > 0 and iou.max() >= thr
        if hit:
            matched.add(int(iou.argmax()) + 1)
        frac = ga[g] / y.size
        b = "small" if frac < bins[0] else ("medium" if frac <= bins[1] else "large")
        out["tp"] += hit; out["fn"] += (not hit); out[f"tp_{b}"] += hit; out[f"fn_{b}"] += (not hit)
    out["fp"] = (npr - 1) - len(matched)
    return out


def boundary_f1(p, y, tol=2):
    k = np.ones((3, 3), np.uint8); kt = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    bp, by = p - cv2.erode(p, k), y - cv2.erode(y, k)
    if bp.sum() == 0 and by.sum() == 0:
        return 1.0
    prec = (bp & cv2.dilate(by, kt)).sum() / max(bp.sum(), 1); rec = (by & cv2.dilate(bp, kt)).sum() / max(by.sum(), 1)
    return float(2 * prec * rec / max(prec + rec, 1e-8))


def image_metrics(p, y, leaf, bins):
    i1 = int(((p == 1) & (y == 1)).sum()); u1 = int(((p == 1) | (y == 1)).sum())
    i0 = int(((p == 0) & (y == 0)).sum()); u0 = int(((p == 0) | (y == 0)).sum())
    r = dict(inter1=i1, union1=u1, inter0=i0, union0=u0, iou=i1 / u1 if u1 else 1.0,
             dice=2 * i1 / max(int(p.sum()) + int(y.sum()), 1) if (p.sum() + y.sum()) else 1.0,
             bf1=boundary_f1(p, y), sev_err=abs(int(p.sum()) - int(y.sum())) / max(float(leaf.sum()), 1.0) * 100,
             lesion_frac=float(y.mean()), pred_frac=float(p.mean()))
    r.update(lesion_match(p, y, bins))
    return r


@torch.no_grad()
def eval_frame(model, frame, cfg, dev, workers=None):
    model.eval(); amp = dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] >= 7
    dl = make_loader(frame, cfg, False, cfg["eval_bs"], shuffle=False, workers=workers)
    rows = []; pool = ThreadPoolExecutor(4)
    for b in dl:
        with torch.autocast("cuda", enabled=amp):
            pred = model(b["img"].to(dev)).argmax(1).cpu().numpy().astype(np.uint8)
        ys = b["mask"].numpy().astype(np.uint8); leaves = b["leaf"].numpy(); idx = b["idx"].numpy()
        res = list(pool.map(lambda k: image_metrics(pred[k], ys[k], leaves[k], cfg["size_bins"]), range(len(pred))))
        for k, r in zip(idx, res):
            r["idx"] = int(k); rows.append(r)
    pool.shutdown()
    if not rows:
        raise ValueError(f"empty evaluation frame ({len(frame)} rows)")
    out = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    for c in ["key", "disease", "bg_share", "n_leaves", "altered"]:
        if c in frame.columns:
            out[c] = frame[c].to_numpy()[out.idx]
    return out


def test_sets(cfg, frames):
    sets = {"plantseg_test": frames["test"]}
    for lv in ["light", "medium", "heavy"]:
        p = wp(cfg, "clutter_test", lv, "index.csv")
        if p.exists():
            f = pd.read_csv(p, dtype={"key": str})
            sets[f"clutter_{lv}"] = f[f.key.isin(set(frames["test"].key))].reset_index(drop=True)
    pd_dir = Path(cfg["paths"].get("plantdoc_seg") or wp(cfg, "plantdoc_seg"))
    if (pd_dir / "masks").exists():
        masks = {p.stem: p for p in (pd_dir / "masks").glob("*.png")}
        rows = [dict(image=str(p), mask=str(masks[p.stem]), key=p.stem, disease=-1)
                for p in sorted((pd_dir / "images").glob("*")) if p.stem in masks]
        if rows:
            sets["plantdoc_seg"] = pd.DataFrame(rows)
    return sets


def load_model(cfg, name, dev):
    info = json.load(open(wp(cfg, "runs", name, "done.json")))
    m = build(info["arch"], info["encoder"], pretrained=False).to(dev)
    m.load_state_dict(torch.load(wp(cfg, "runs", name, "best.pt"), map_location=dev)); m.eval()
    return m, info


def eval_run(cfg, name, frames, dev, workers=None):
    out_dir = wp(cfg, "results", "per_image"); out_dir.mkdir(parents=True, exist_ok=True)
    m, info = load_model(cfg, name, dev)
    for tname, fr in test_sets(cfg, frames).items():
        p = out_dir / f"{name}__{tname}.csv"
        if p.exists():                                    # keep only complete files (an old half-written file is redone)
            try:
                ok = len(pd.read_csv(p)) == len(fr)
            except Exception:
                ok = False
            if not ok:
                log(f"[{name}] incomplete result file {p.name}: evaluating again"); p.unlink()
        if not p.exists():
            atomic_to_csv(eval_frame(m, fr, cfg, dev, workers=workers), p, index=False)
    (wp(cfg, "results", "evaldone")).mkdir(parents=True, exist_ok=True)
    atomic_write_text(wp(cfg, "results", "evaldone", name), str(time.time()))
    del m
    if dev.type == "cuda":
        torch.cuda.empty_cache()

# --------------------------------------------------------------------------------------- worker
def worker(work, gpu):
    cfg = load_cfg(work)
    if keep_awake():
        log("Windows sleep and screen-off blocked while training runs")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    frames = load_split_frames(cfg); bank = make_bank(frames["train"], cfg)
    log(f"worker gpu={gpu} device={torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'} | "
        f"train {len(frames['train'])} val {len(frames['val'])} test {len(frames['test'])} | quick={cfg['quick']}")
    deadline = cfg.get("deadline")
    while True:
        if deadline and time.time() > deadline - 900:
            log("session time budget reached; exiting (state saved, rerun the notebook to continue)"); return
        status, x = next_job(cfg, gpu)
        if status == "job":
            j = x
            try:
                log(f"START {j['name']}")
                st = train_run(cfg, j, frames, bank, dev, deadline)
                if st == "done":
                    if not eval_done(cfg, j["name"]):
                        eval_run(cfg, j["name"], frames, dev)
                    log(f"DONE {j['name']}")
                    (wp(cfg, "runs", j["name"]) / "error.txt").unlink(missing_ok=True)   # an old crash of this run is now fixed
                else:
                    release(cfg, j["name"]); return
            except Exception:
                log(f"ERROR in {j['name']}:\n{traceback.format_exc()}")
                wp(cfg, "runs", j["name"]).mkdir(parents=True, exist_ok=True)
                wp(cfg, "runs", j["name"], "error.txt").write_text(traceback.format_exc())
                release(cfg, j["name"]); raise
            release(cfg, j["name"])
        elif status == "wait":
            log(f"waiting: {x}"); time.sleep(60)
        else:
            log(f"{status.upper()}: {x}"); return


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "worker":
        worker(sys.argv[2], int(sys.argv[3]))