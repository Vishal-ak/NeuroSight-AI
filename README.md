# 🧠 NeuroSight AI

**See beyond the scan.** An explainable deep-learning workspace for brain MRI image classification, model evaluation and dataset quality analysis.

NeuroSight AI is an educational/research prototype that explores four image labels: **glioma, meningioma, pituitary tumor and no tumor**. It combines a PyTorch training pipeline with a Streamlit application for image inference, model analytics, Grad-CAM visualizations and dataset auditing.

> **Safety:** This is not a medical device and is not validated for clinical use. Never use it to diagnose, rule out disease, or guide treatment. Model scores are not calibrated clinical probabilities.

## Run the application

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard has four pages: MRI Prediction, Model Analytics, Grad-CAM Explainability and Dataset Audit. Pages read trained checkpoints and generated reports. **No performance numbers are fabricated**; analytics remains empty until actual model evaluation files exist.

## Repository layout

```text
NeuroSight-AI/
├── app.py
├── pages/                  # Streamlit dashboard pages
├── src/                    # Models, training, inference and explainability
├── scripts/                # Dataset audit and report generation
├── tests/                  # Automated tests
├── notebooks/              # EDA and experiment notebooks
├── configs/                # Experiment configuration
├── data/                   # Raw data location (not committed)
├── models/                 # Trained checkpoints (not committed)
├── outputs/                # Generated reports and metrics
├── .streamlit/             # Dashboard theme
├── requirements.txt
└── README.md
```

## Dataset setup

Extract the provided dataset into:

```text
data/raw/Brain_Tumor_MRI_Dataset/
├── Training/{glioma,meningioma,notumor,pituitary}/
└── Testing/{glioma,meningioma,notumor,pituitary}/
```

Run the audit:

```bash
python scripts/audit_dataset.py --data-dir data/raw/Brain_Tumor_MRI_Dataset
```

The audit verifies image readability, records dimensions/modes, calculates SHA-256 hashes, identifies exact duplicates and flags identical bytes with conflicting labels. For an exact duplicate group found in both original folders, the retained copy is assigned to the test partition so that identical bytes do not appear in both partitions.

**Audit of the archive supplied for this project:** 7,023 readable images, 6,726 unique SHA-256 hashes, 194 exact-duplicate groups and 79 groups crossing the original train/test folders. Hashing does not detect visually similar images or establish patient-level independence. Treat resulting performance as an image-level benchmark, not patient-level or clinical validation.

## Train models

```bash
# Baseline CNN
python -m src.train --data-dir data/raw/Brain_Tumor_MRI_Dataset --model cnn --epochs 15

# Transfer-learning models
python -m src.train --data-dir data/raw/Brain_Tumor_MRI_Dataset --model resnet18 --epochs 15
python -m src.train --data-dir data/raw/Brain_Tumor_MRI_Dataset --model mobilenet_v2 --epochs 15
```

Training creates a stratified validation split from the deduplicated training partition, selects the best checkpoint by validation macro-F1, and evaluates that checkpoint on the deduplicated test partition. Metrics are written to `outputs/<model>/test_metrics.json`; the selected checkpoint is saved to both `outputs/<model>/best_model.pt` and `models/<model>/best_model.pt` for dashboard inference.

For offline runs, add `--no-pretrained` to avoid downloading ImageNet weights for ResNet18/MobileNetV2. Use a GPU where available; reduce `--batch-size` if memory is limited.

## Tests

```bash
python -m pytest -q
```

Three starter tests are included. They have not yet been executed in a clean target environment; run the suite and verify the result before treating the repository as validated.

## Google Colab

Clone this repository in Colab, install dependencies, mount/upload the dataset, run the audit and training commands above, then launch the dashboard with `streamlit run app.py`. Persist checkpoints and generated outputs in Drive if needed. Do not commit raw MRI images or model weights to GitHub.

## External dataset experiment

The Cheng et al. Figshare brain tumor collection is a possible three-tumor-class research benchmark: <https://figshare.com/articles/dataset/brain_tumor_dataset/1512427>. It does not provide the same four-class task. Check provenance and exact/perceptual overlap before calling it independent external validation; preserve patient identifiers and use patient-disjoint splits when available.

## Project status

- [x] Dataset audit and duplicate-aware manifest generation
- [x] Streamlit dashboard and model inference integration
- [x] CNN, ResNet18 and MobileNetV2 training implementations
- [x] Grad-CAM integration
- [x] Model analytics and dataset audit pages
- [x] Initial local smoke tests
- [ ] Train all model variants and record real results
- [ ] Inspect perceptual duplicates and dataset provenance
- [ ] Validate on an independent patient-disjoint dataset
- [ ] Deploy after selecting and reviewing a trained checkpoint

## License

Project code is released under the MIT License. Dataset images and third-party pretrained weights remain subject to their respective licenses and terms.
