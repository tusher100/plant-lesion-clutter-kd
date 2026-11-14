r"""
INT8 TFLite export + accuracy check for Paper 1 (replaces notebook cell 9's onnx2tf step, which fails with an
old onnx2tf: "This file contains pickled (object) data"). Runs in its OWN conda env, so the training env
(cropai) is never touched.

One-time setup (Anaconda Prompt):
    conda create -n tfl python=3.11 -y
    conda activate tfl
    pip install onnx2tf onnx onnx-graphsurgeon sng4onnx onnxruntime tensorflow tf_keras ai-edge-litert opencv-python pandas psutil

Run (after the notebook's Run All has written export\*.onnx and export\calib_nhwc_0to1.npy):
    conda activate tfl
    python int8_export.py "<P1_ROOT>\paper1_runs_v2"
    python int8_export.py "<P1_ROOT>\paper1_runs_nlb"

Output: <work>\export\tflite_<run>\<run>_integer_quant.tflite  (the file for the phone)
        <work>\results\table_int8.csv   FP32 (ONNX) vs INT8 (TFLite) mIoU on the test set, same preprocessing
"""
import json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, cv2

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def convert(onnx, od, calib):
    exe = shutil.which("onnx2tf") or str(Path(sys.executable).parent / "Scripts" / "onnx2tf.exe")
    cmd = [exe, "-i", str(onnx), "-o", str(od), "-tb", "tf_converter", "-oiqt", "-cind", "input", str(calib),
           str([[[MEAN.tolist()]]]), str([[[STD.tolist()]]])]
    t0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True)
    tfl = [p for p in od.glob("*_integer_quant.tflite") if "full_integer" not in p.name and "int16" not in p.name]
    if not tfl:
        print("  conversion FAILED:\n", (r.stdout + r.stderr)[-2000:]); return None
    print(f"  converted in {time.time() - t0:.0f} s -> {tfl[0].name} ({tfl[0].stat().st_size / 1e6:.1f} MB)")
    return tfl[0]


def miou(pred_fn, test, size):
    inter, union = np.zeros(2), np.zeros(2)
    for r in test.itertuples():
        img = cv2.cvtColor(cv2.imread(r.cimg), cv2.COLOR_BGR2RGB); y = (cv2.imread(r.cmsk, 0) > 0).astype(np.uint8)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR); y = cv2.resize(y, (size, size), interpolation=cv2.INTER_NEAREST)
        p = pred_fn(((img / 255.0 - MEAN) / STD).astype(np.float32)[None])
        for c in (0, 1):
            inter[c] += ((p == c) & (y == c)).sum(); union[c] += ((p == c) | (y == c)).sum()
    return float((inter / np.maximum(union, 1)).mean() * 100)


def main(work):
    work = Path(work); exp = work / "export"; cfg = json.load(open(work / "config.json")); size = cfg["img_size"]
    calib = exp / "calib_nhwc_0to1.npy"
    assert calib.exists(), f"{calib} missing: run the notebook (Run All) first"
    df = pd.read_csv(work / "splits" / "split.csv"); cache = Path(cfg["paths"]["cache"])
    test = df[df.split == "test"].copy()
    test["cimg"] = [str(cache / "img" / f"{u:05d}.jpg") for u in test.uid]; test["cmsk"] = [str(cache / "msk" / f"{u:05d}.png") for u in test.uid]
    import onnxruntime as ort, tensorflow as tf
    rows = []
    for onnx in sorted(exp.glob("*.onnx")):
        run = onnx.stem
        if not ("A7_full" in run or "A0_student" in run or "direct" in run):
            continue                                                   # the teacher is not meant for the phone
        print(run)
        sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        fp32 = miou(lambda x: sess.run(None, {"input": x.transpose(0, 3, 1, 2)})[0][0].argmax(0), test, size)
        tfl = convert(onnx, exp / f"tflite_{run}", calib)
        if tfl is None:
            rows.append(dict(run=run, fp32_miou=round(fp32, 2))); continue
        it = tf.lite.Interpreter(model_path=str(tfl)); it.allocate_tensors()
        inp, out = it.get_input_details()[0], it.get_output_details()[0]

        def pred(x):
            if inp["shape"][1] == 3:
                x = x.transpose(0, 3, 1, 2)
            if inp["dtype"] != np.float32:                             # int8 input: quantise with the model's own scale
                sc, zp = inp["quantization"]; x = np.clip(np.round(x / sc + zp), -128, 127).astype(inp["dtype"])
            it.set_tensor(inp["index"], x); it.invoke(); o = it.get_tensor(out["index"])[0]
            return o.argmax(-1) if o.shape[-1] == 2 else o.argmax(0)

        i8 = miou(pred, test, size)
        rows.append(dict(run=run, fp32_miou=round(fp32, 2), int8_miou=round(i8, 2), drop=round(fp32 - i8, 2),
                         fp32_onnx_MB=round(onnx.stat().st_size / 1e6, 2), int8_MB=round(tfl.stat().st_size / 1e6, 2),
                         ok="PASS" if fp32 - i8 < 1.5 else "drop > 1.5: report it, or use quantisation-aware training"))
        print("  ", rows[-1])
    t = pd.DataFrame(rows); (work / "results").mkdir(exist_ok=True); t.to_csv(work / "results" / "table_int8.csv", index=False)
    print("\n", t.to_string(index=False), "\nsaved", work / "results" / "table_int8.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
    else:
        main(sys.argv[1])
