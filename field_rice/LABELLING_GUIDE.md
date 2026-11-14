# Labelling guide: 60 rice field photos (Paper 1 real-field test)

Time needed: about 2 hours for all 60 (2 minutes per photo).

## IMPORTANT: the labels in `labels/` are machine-made DRAFTS
Every JSON in `labels/` was made automatically: the leaf is a band around a line placed by eye, and
lesions are the brown/yellow/straw pixels inside it. They are often wrong (band off the leaf, grass
included, spots missed, healthy pale tissue marked). Open every photo in LabelMe and fix it:
move or redraw the `leaf` polygon onto the real leaf edge, delete wrong `lesion` polygons, add missed ones.
Do not report results on uncorrected drafts: the test would then measure agreement with the colour rule,
not with a person. `preview_drafts.jpg` shows all drafts (blue = leaf, red = lesion).
Weakest drafts (redraw from scratch): Bacterial_leaf_blight_01, 02, 11, 12, 13, 15 and Sheath_blight_04 to 15
(in the sheath-blight clumps pick one tiller you can follow and draw that as the leaf).
Expected time: about 30-40 minutes.

## Tool
LabelMe (free): `pip install labelme`, then run `labelme images --output labels --nodata --autosave`.
Every photo you save becomes `labels/<photo name>.json`. Keep the file names as they are.

## What to draw (two labels only, lowercase)

1. **leaf**: one polygon around the ONE leaf the photo is about (the leaf in focus, usually in the centre).
   - Follow the leaf edge; 15-30 points is enough.
   - If the target leaf is partly hidden by grass, a finger or another leaf, draw only the part you can see.
   - Do not include other leaves, grass, pots or soil.
2. **lesion**: one polygon for each diseased spot or streak ON THAT LEAF.
   - Include brown, grey, yellow-orange dead tissue and the dark spot margins.
   - Do not include the plain yellow halo around a spot if the tissue is still green-yellow and alive.
   - Spots smaller than about 3 mm can be skipped.
   - Blight that runs from the tip: draw one polygon around the whole dead part.
   - Lesions on other leaves in the background: do NOT label them (the model is judged on the target leaf; spots on other leaves are counted as neither right nor wrong for severity, but they will count against the model in the whole-photo IoU, which is what we want to measure).

## When to skip a photo
- No visible lesion, or the photo is too blurred to see where lesions end: close it without saving.
- Sheath-blight photos where the lesions are on the stem, not on a leaf: still draw them, and draw the
  stem/sheath part in focus as **leaf**.

## Check before you finish
- Each saved JSON has exactly one `leaf` and at least one `lesion`.
- Open 5 random photos again and check the polygons sit on the right places.

## Optional (makes the result stronger)
Ask a second person (or an agronomist) to label 15 of the photos (photos `_01` to `_04` of each disease)
in a separate folder `labels_b/`. This gives a human-vs-human agreement figure to compare the model with.

## After labelling
Put the folder on the PC like this and run the script:

```
<P1_ROOT>\field_test\images\        (60 jpg)
<P1_ROOT>\field_test\labels\        (your json files)
<P1_ROOT>\field_test\field_images.csv
```

```
conda activate cropai
cd <repo>
python field_eval.py --work "<P1_ROOT>\paper1_runs_v2" --field "<P1_ROOT>\field_test"
```

It runs in a few minutes on the GPU (CPU also works). Send me the `results` folder.
