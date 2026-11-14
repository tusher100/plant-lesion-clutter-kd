# Few-shot adaptation to rice field photographs (Section V-G)

60 smartphone photographs of rice (bacterial leaf blight, brown spot, blast, sheath blight; 15 each), one per burst
of near-identical shots, with the target leaf and its lesions annotated as polygons (LabelMe).
The photographs come from an unpublished collection and are not included in this repository.

| Script | Purpose |
|---|---|
| `select_field_images.py` | picks one photo per burst group (dHash), resizes to 2048 px, removes all metadata (incl. GPS) |
| `LABELLING_GUIDE.md` | how the `leaf` and `lesion` polygons were drawn |
| `field_eval.py` | runs trained models on labelled photos: lesion IoU, in-leaf lesion IoU, severity MAE / kappa / CCC |
| `finetune_field.py` | 3-fold cross-validation (stratified by disease), fine-tunes from PlantSeg checkpoints or ImageNet |

```bash
cd field_rice
python finetune_field.py --work <run folder> --field <folder with images/, labels/, field_images.csv> --shots 2 5 10
```
Fixed budget of 300 iterations, no early stopping; the training subsets are identical for every initialization.
Results: `results/rice_fewshot_*.csv`.
