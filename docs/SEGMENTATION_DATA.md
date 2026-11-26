# Segmentation dataset contract

## Why a separate dataset is needed

The current NeuroSight four-class dataset supplies image-level labels (glioma, meningioma, pituitary and no-tumor). It does not provide a binary or multi-class pixel mask for each image. A classification label cannot be converted into a reliable tumor segmentation mask.

Do not create synthetic masks from class labels or Grad-CAM maps and treat them as ground truth.

## Required manifest

Create `data/segmentation/manifest.csv` with these columns:

| Column | Meaning |
|---|---|
| image_path | Path to MRI image |
| mask_path | Path to aligned expert-annotated mask |
| split | train, val or test |
| patient_id | Stable patient identifier where available |
| source | Dataset/source identifier |
| mask_type | Binary tumor mask or documented tumor subregion labels |

Example:

```csv
image_path,mask_path,split,patient_id,source,mask_type
images/p001_slice01.png,masks/p001_slice01.png,train,p001,source_a,binary
images/p002_slice02.png,masks/p002_slice02.png,val,p002,source_a,binary
images/p003_slice01.png,masks/p003_slice01.png,test,p003,source_a,binary
```

The example is illustrative only; it does not represent actual data.

## Quality checks

- Image and mask must refer to the same slice and be spatially aligned.
- Masks must have documented semantics and a known pixel-value mapping.
- Binary masks should map background to 0 and tumor to 1 after loading.
- Patient IDs must not cross train/validation/test splits.
- If patient IDs are unavailable, disclose the limitation; do not claim patient-independent validation.
- Keep source licensing, attribution and original metadata.
- Never publish patient-identifiable images or metadata.

## Evaluation

Report Dice coefficient and Intersection over Union (IoU), both overall and per case. Also inspect empty-mask cases, false-positive regions, false-negative regions and overlay quality. Do not report only a single aggregate score.
