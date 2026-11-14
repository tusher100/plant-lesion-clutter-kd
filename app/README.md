# Leaf Lesion Scan: offline field demo for Paper 1

Takes a photo of a diseased leaf, runs the distilled student (TFLite) **on the phone, without internet**, shows
the lesion mask, the severity (lesion share of leaf area) and its class (0-5 / 5-25 / >25 %), and keeps a
**field log**: every saved photo plus a CSV row. 

## 1. Add the model (required)
Copy the Lite0 full-INT8 model from the PC into `assets/` and rename it to `model.tflite`:

    <P1_ROOT>\paper1_runs_v2\export\tflite_tf_efficientnet_lite0__A7_full__s0\
        tf_efficientnet_lite0__A7_full__s0_integer_quant.tflite   ->   assets/model.tflite

The model file is not in the repository; download it from the Releases page or export your own. Any other exported variant
(dynamic range, FP16, MobileNetV3) also works: the app reads the input/output layout and type from the model.

## 2. Run
    flutter pub get
    flutter run --release          # phone connected with USB debugging

Tested with Flutter 3.47.9 / Dart 3.13.

## 3. Field session
1. Camera -> photograph one diseased leaf as a farmer would (other leaves, soil, shade, hand in view are fine).
2. Check the overlay, optionally type a note (crop, disease, place), press **Save to field log**.
3. At the end, press the share icon (top right) -> send the CSV and photos to Google Drive or e-mail.
4. Label the lesions in the saved photos (CVAT / LabelMe) -> the evaluation script (field_rice/field_eval.py) compares the app's
   severity and the models' masks with your labels.

## How it matches the training pipeline
- The whole photo is resized to 320 x 320 (aspect ratio not kept), ImageNet mean/std normalisation, as in training.
- Leaf = excess-green index + Otsu threshold on the resized image, united with the predicted lesion pixels.
- The mask is stretched back over the photo, which inverts the resize exactly.
- Inference runs on 4 CPU threads (about 18 ms for Lite0 INT8 on a Snapdragon 8s Gen 3).

Severity classes are illustrative, not a spraying recommendation.
