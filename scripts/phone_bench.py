r"""
Phone speed test for Paper 1 (TFLite benchmark_model over USB). Measures on-device latency and peak memory
of the exported models on the CPU (4 threads and 1 thread) and on the phone GPU.

Needs: Android phone with USB debugging ON, connected by USB (accept the "Allow USB debugging?" prompt on the
phone), and Google's adb (Android SDK platform-tools, free):
    https://developer.android.com/tools/releases/platform-tools  -> "Download SDK Platform-Tools for Windows"
    unzip it to C:\platform-tools   (so that C:\platform-tools\adb.exe exists)

Run (Anaconda Prompt, tfl env; the PC GPU is not used, so training can keep running):
    conda activate tfl
    python phone_bench.py check        # 1) is the phone visible? which phone / chip / Android?
    python phone_bench.py run          # 2) about 15-30 min; keep the phone unlocked, screen on, not charging hot

Output: <P1_ROOT>\paper1_runs_v2\results\table_phone.csv  (+ phone_bench_raw.txt with the full logs)
"""
import re, shutil, subprocess, sys, time, urllib.request
import os
from pathlib import Path
import pandas as pd

WORK = Path(os.environ.get("P1_ROOT", ".")) / r"paper1_runs_v2"
EXP = WORK / "export"
HERE = Path(__file__).resolve().parent
BENCH_URL = ("https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/"
             "latest/android_aarch64_benchmark_model")
REMOTE = "/data/local/tmp/p1bench"
# (run, tflite variant suffix, label)
MODELS = [
    ("mobilenetv3_small_100__A7_full__s0", "_dynamic_range_quant", "MobileNetV3 + A7, dynamic-range"),
    ("mobilenetv3_small_100__A7_full__s0", "_integer_quant_with_int16_act", "MobileNetV3 + A7, INT8 w / INT16 act"),
    ("mobilenetv3_small_100__A7_full__s0", "_float16", "MobileNetV3 + A7, FP16"),
    ("mobilenetv3_small_100__A0_student__s0", "_dynamic_range_quant", "MobileNetV3 student alone, dynamic-range"),
    ("tf_efficientnet_lite0__A7_full__s0", "_integer_quant", "Lite0 + A7, full INT8"),
    ("tf_efficientnet_lite0__A7_full__s0", "_float16", "Lite0 + A7, FP16"),
    ("D-dlv3p_mobilenetv2__direct__s0", "_integer_quant", "DeepLabV3+ MobileNetV2, full INT8"),
]
CONFIGS = [("CPU 4 threads", ["--num_threads=4"]), ("CPU 1 thread", ["--num_threads=1"]), ("GPU", ["--use_gpu=true"])]


def adb_path():
    for c in [shutil.which("adb"), r"C:\platform-tools\adb.exe", str(HERE / "platform-tools" / "adb.exe"),
              str(Path.home() / "AppData/Local/Android/Sdk/platform-tools/adb.exe")]:
        if c and Path(c).exists():
            return c
    sys.exit("adb not found: download platform-tools and unzip to C:\\platform-tools (see the top of this file)")


def adb(*a, timeout=900):
    r = subprocess.run([adb_path(), *a], capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    return r.stdout + r.stderr


def phone_info():
    p = lambda k: adb("shell", "getprop", k).strip()
    return dict(phone=f"{p('ro.product.manufacturer')} {p('ro.product.model')}", chip=p("ro.soc.model") or p("ro.board.platform"),
                android=p("ro.build.version.release"), abi=p("ro.product.cpu.abi"))


def check():
    out = adb("devices")
    print(out)
    if "\tdevice" not in out:
        sys.exit("No phone ready. 'unauthorized' -> unlock the phone and tap Allow on the USB debugging prompt; "
                 "nothing listed -> try another cable/port and set USB mode to 'File transfer'.")
    print(phone_info())
    missing = [f"{r}{v}" for r, v, _ in MODELS if not (EXP / f"tflite_{r}" / f"{r}{v}.tflite").exists()]
    print("models found:", len(MODELS) - len(missing), "/", len(MODELS), ("missing: " + ", ".join(missing)) if missing else "")
    print("\nOK -> python phone_bench.py run")


def parse(log):
    g = lambda pat: (float(m.group(1)) if (m := re.search(pat, log)) else None)
    avg = g(r"Inference \(avg\):\s*([\d.e+]+)")
    mn = g(r"count=\d+ first=[\d.]+ curr=[\d.]+ min=([\d.]+)")
    runs = re.findall(r"count=\d+ first=[\d.]+ curr=[\d.]+ min=[\d.]+ max=[\d.]+ avg=([\d.e+]+) std=([\d.]+)", log)
    std = float(runs[-1][1]) if runs else None
    mem = g(r"Overall peak memory footprint \(MB\) via periodic monitoring:\s*([\d.]+)")
    return dict(latency_ms=round(avg / 1000, 2) if avg else None, min_ms=round(mn / 1000, 2) if mn else None,
                std_ms=round(std / 1000, 2) if std else None, peak_mem_MB=mem)


def run():
    check()
    bench = HERE / "android_aarch64_benchmark_model"
    if not bench.exists():
        print("downloading benchmark_model (Google TFLite tool, ~10 MB) ...")
        urllib.request.urlretrieve(BENCH_URL, bench)
    adb("shell", "mkdir", "-p", REMOTE)
    adb("push", str(bench), f"{REMOTE}/benchmark_model"); adb("shell", "chmod", "+x", f"{REMOTE}/benchmark_model")
    info = phone_info(); rows = []; raw = []
    for r, v, label in MODELS:
        f = EXP / f"tflite_{r}" / f"{r}{v}.tflite"
        if not f.exists():
            print("missing, skipped:", f.name); continue
        adb("push", str(f), f"{REMOTE}/m.tflite")
        for cname, flags in CONFIGS:
            time.sleep(20)                                           # let the phone cool between runs
            log = adb("shell", f"{REMOTE}/benchmark_model", f"--graph={REMOTE}/m.tflite", "--warmup_runs=30",
                      "--num_runs=100", "--report_peak_memory_footprint=true", *flags)
            res = parse(log)
            ok = res["latency_ms"] is not None and "fail" not in log.lower().split("inference timings")[-1][:200]
            gpu_note = ""
            if cname == "GPU":
                applied = re.search(r"Explicitly applied GPU delegate, and the model graph will be (completely|partially) executed", log)
                gpu_note = f"GPU {applied.group(1)}" if applied else "GPU not applied (ran on CPU or failed)"
            rows.append(dict(model=label, file=f.name, size_MB=round(f.stat().st_size / 1e6, 2), config=cname, **res,
                             note=gpu_note if ok else "failed: see phone_bench_raw.txt", **info))
            raw.append(f"===== {label} | {cname}\n{log}\n")
            print(f"{label:45s} {cname:14s} {res['latency_ms']} ms  peak {res['peak_mem_MB']} MB  {gpu_note}")
    out = WORK / "results"; out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "table_phone.csv", index=False)
    (out / "phone_bench_raw.txt").write_text("\n".join(raw), encoding="utf-8")
    adb("shell", "rm", "-rf", REMOTE)
    print("\nsaved", out / "table_phone.csv")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"check": check, "run": run}.get(cmd, lambda: print(__doc__))()
