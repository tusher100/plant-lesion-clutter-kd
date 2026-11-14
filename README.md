# What Makes On-Device Plant Lesion Segmentation Robust to Field Clutter?

Code for the paper **"What Makes On-Device Plant Lesion Segmentation Robust to Field Clutter? Augmentation,
Distillation and Clean-View Targets"** (submitted to *IEEE Transactions on Industrial Informatics*).

We train compact U-Net students (MobileNetV3-Small, EfficientNet-Lite0) for binary lesion segmentation, distilled
from a SegFormer-B4 teacher, and study what makes them robust to field clutter (overlapping leaves, soil, shadows,
glare). Main findings on PlantSeg:

- clutter augmentation with occlusion-aware labels supplies the robustness (heavy clutter 61.4 → 70.2 mIoU, no teacher);
- distillation on clean images supplies clean accuracy (+1.7 mIoU); the simple combination matches every more
  elaborate variant;
- distilling the teacher's prediction on the cluttered view is harmful (−3.1); a clean-view target removes the harm;
- an INT8 EfficientNet-Lite0 student (5.8 MB) runs in 18.3 ms on a smartphone CPU.

## Repository layout

```
p1lib/          p1lib.py      data, clutter generator, models, losses, training queue, evaluation
                p1lib_rev.py  same library + the factorial configurations (R1-R6) used in the paper
notebooks/      Paper1_PlantSeg.ipynb, Paper1_NLB.ipynb   end-to-end runs (data prep, training, evaluation, export)
scripts/        rev1_ablation.py   factorial study (Tables IV-V)
                rev34_eval.py      held-out occluders, unseen corruptions, severity agreement (Table VIII)
                nlb_wacv_prepare.py NLB maize masks + tiles, split by photograph
                int8_export.py     ONNX -> TFLite (full INT8, FP16, dynamic range) + accuracy check
                phone_bench.py     on-device latency with TFLite benchmark_model over adb
                qual_figure.py     qualitative figure
field_rice/     few-shot adaptation to rice field photographs (Section V-G, Table X)
app/            Flutter app: offline lesion mask + severity on the phone
results/        CSV tables behind the paper's results
```

## Installation

```bash
conda env create -f environment.yml
conda activate cropai
```
Tested on Windows 11, one NVIDIA GTX 1080 Ti (11 GB), PyTorch 2.7.1 + CUDA 11.8.
TFLite export and the phone benchmark use a separate environment (see `scripts/int8_export.py`).

## Data

| Dataset | Source | Notes |
|---|---|---|
| PlantSeg | Wei et al., first public release, arXiv:2409.04038 | 11,458 images; we build our own leak-free split (perceptual-hash groups, 70/15/15) |
| NLB maize | Prashanth et al., WACV 2024; Wiesner-Hanks et al., 2018 | per-lesion masks merged per image; `scripts/nlb_wacv_prepare.py` builds the tiles |

Datasets are not redistributed here; download them from the original sources. Set the folder that holds
`Dataset/` and the run folders:

```bash
set P1_ROOT=D:\path\to\workspace        # Windows   (export P1_ROOT=... on Linux)
```

## Reproducing the paper

1. **Main runs (Tables VI, VII, IX):** open `notebooks/Paper1_PlantSeg.ipynb` (or `Paper1_NLB.ipynb`), set the
   paths in cell 1 and run all. Cell 0 writes `p1lib.py`; training runs as a resumable job queue
   (`python p1lib.py worker <WORK> <gpu>`). Every student is trained with three seeds.
2. **Factorial study (Tables IV, V):** `cd scripts && python rev1_ablation.py run`, then `... summary`.
3. **Held-out robustness and severity (Table VIII):** `python rev34_eval.py build`, `eval`, `summary`.
4. **Deployment (Table IX):** `python int8_export.py <work>` in the `tfl` environment, then `python phone_bench.py run`.
5. **Rice few-shot adaptation (Table X):** see `field_rice/README.md`.

The CSV files in `results/` are the outputs of these steps.

## Pretrained models

Trained weights (`best.pt`) and TFLite models are published on the Releases page of this repository
(links added at release).

## Phone app

See `app/README.md`. Flutter 3.47.9, `tflite_flutter` 0.12.1; place an exported model at `app/assets/model.tflite`.

## Citation

Citation details will be added on publication.

## License

Code: MIT (see `LICENSE`). Datasets keep their original licenses.
