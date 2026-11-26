# 🧠 NeuroSight AI

**See beyond the scan.** An explainable deep-learning workspace for brain MRI image classification, model evaluation, dataset quality analysis and experimental segmentation.

NeuroSight AI is an educational/research prototype for four image labels: **glioma, meningioma, pituitary tumor and no tumor**. It is not a medical device and is not validated for clinical use. Never use it to diagnose, rule out disease or guide treatment. Model scores are not calibrated clinical probabilities.

## Run the application

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard includes MRI Prediction, Model Analytics, Grad-CAM Explainability, Dataset Audit and Tumor Segmentation. Analytics reads real metrics files only; it does not invent results.

## Dataset setup — important

Use only the original image dataset with this structure:

```text
data/raw/Brain_Tumor_MRI_Dataset/
├── Training/{glioma,meningioma,notumor,pituitary}/
└── Testing/{glioma,meningioma,notumor,pituitary}/
```

**Do not use `Output_images` or any previous-run prediction/visualization folder as input data.** The audit and training pipeline explicitly rejects a data root named `Output_images` and rejects manifests containing paths from that folder. `Output_images` is historical output, not a source dataset.

Run the audit against the original dataset:

```bash
python scripts/audit_dataset.py --data-dir data/raw/Brain_Tumor_MRI_Dataset
```

The audit verifies readability, records image dimensions/modes, calculates SHA-256 hashes, identifies exact duplicates and flags identical bytes with conflicting labels. If an exact duplicate occurs in both original partitions, its retained copy is assigned to the test partition so identical bytes do not occur in both partitions. Exact hashing does not detect perceptual duplicates or establish patient-level independence.

## Train all three classifiers

Run each command with the same dataset manifest and seed. They create independent checkpoints and save validation/test predictions for fair comparison.

```bash
python -m src.train --data-dir data/raw/Brain_Tumor_MRI_Dataset --model cnn --epochs 15 --seed 42
python -m src.train --data-dir data/raw/Brain_Tumor_MRI_Dataset --model resnet18 --epochs 15 --seed 42
python -m src.train --data-dir data/raw/Brain_Tumor_MRI_Dataset --model mobilenet_v2 --epochs 15 --seed 42
```

For offline runs, add `--no-pretrained` to avoid downloading ImageNet weights for ResNet18/MobileNetV2. Use a GPU where available; reduce `--batch-size` if memory is limited.

Each run saves:
- `models/<model>/best_model.pt` and `outputs/<model>/best_model.pt`: checkpoint selected by validation macro-F1.
- `outputs/<model>/training_history.json`: per-epoch train/validation loss, accuracy, validation macro-F1 and hyperparameters.
- `outputs/<model>/validation_predictions.npz`: validation labels, probabilities and paths.
- `outputs/<model>/test_predictions.npz`: held-out test labels, probabilities and paths.
- `outputs/<model>/test_metrics.json`: test accuracy, macro/weighted F1, per-class precision/recall/sensitivity/specificity, confusion matrix, ROC-AUC/PR-AUC where defined, approximate inference throughput and checkpoint size.

## Tune ensemble and select overall best

After all three model runs complete:

```bash
python -m src.ensemble
```

The ensemble script:
1. Checks that all models produced predictions for exactly the same validation and test samples in the same order.
2. Searches non-negative probability weights that sum to one, using validation macro-F1 only.
3. Evaluates the selected weighted ensemble on the held-out test predictions.
4. Compares the ensemble with the individual models using validation macro-F1.
5. Writes the winning individual checkpoint or ensemble bundle to `models/overall_best/best_model.pt`.

Outputs include:
- `models/ensemble/ensemble_config.json`
- `models/overall_best/selection.json`
- `outputs/ensemble/test_metrics.json`
- `outputs/comparison/model_comparison.csv` (validation/test scores, inference estimate, checkpoint size and winner flag)
- `outputs/comparison/comparison_report.json`

The MRI Prediction page defaults to `models/overall_best/best_model.pt` when it exists. You can still select CNN, ResNet18 or MobileNetV2 individually. The Model Analytics page displays available evaluation metrics and training curves.

**Selection discipline:** weights and winner are selected using validation data. The test partition is held out for final reporting and must not be used to tune weights or choose the winner. The selected ensemble's validation score may be optimistic because weights are tuned on that validation set; report the untouched test result and consider a separate external/patient-disjoint validation before making strong claims.

## Recommended storage layout

```text
models/
├── cnn/best_model.pt
├── resnet18/best_model.pt
├── mobilenet_v2/best_model.pt
├── ensemble/{best_model.pt,ensemble_config.json}
└── overall_best/
    ├── best_model.pt
    └── selection.json

outputs/
├── cnn/{training_history.json,validation_predictions.npz,test_predictions.npz,test_metrics.json}
├── resnet18/{...}
├── mobilenet_v2/{...}
├── ensemble/test_metrics.json
└── comparison/{model_comparison.csv,comparison_report.json}
```

Model weights and generated outputs are excluded from Git by `.gitignore`. Back them up locally or in Google Drive; do not commit raw MRI images or checkpoints to a public repository.

## Tests

```bash
python -m pytest -q
```

The CI workflow runs the automated tests on pushes and pull requests. A passing local test run or CI run must be verified before claiming the pipeline is validated.

## U-Net segmentation

The repository also contains a U-Net architecture and training scaffold. The four-class classification dataset has image-level labels, not pixel-level tumor masks, so it cannot train U-Net. Use a separate expert-annotated segmentation dataset and patient-disjoint splits where identifiers exist. See [docs/SEGMENTATION_DATA.md](docs/SEGMENTATION_DATA.md).

## Existing dataset audit

The archive previously supplied for this project was reported as 7,023 readable images, 6,726 unique SHA-256 hashes, 194 exact-duplicate groups and 79 groups crossing the original train/test folders. These are audit facts, not model performance. Hashing does not detect visually similar images or prove patient-level independence.

## License

Project code is released under the MIT License. Dataset images and third-party pretrained weights remain subject to their respective licenses and terms.
