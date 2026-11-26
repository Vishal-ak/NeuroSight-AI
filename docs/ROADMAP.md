# NeuroSight AI — Development Roadmap

**Direction:** Research-grade ML + deployable application.
**Product principle:** Every displayed metric must come from a reproducible run. No fake scores, no untrained-model demos presented as results.

## Phase 0 — Make the current classification system reliable (first)

- [ ] Re-run the dataset audit on the exact archive and preserve the audit summary.
- [ ] Validate the manifest, class mapping, split counts and exact duplicate handling.
- [ ] Run unit tests and add tests for model output shapes, inference and Grad-CAM.
- [ ] Train the baseline CNN on a GPU; save configuration, seed, environment and logs.
- [ ] Evaluate the selected checkpoint once on the held-out image-level test set.
- [ ] Verify Streamlit prediction and analytics pages against real generated artifacts.
- [ ] Fix any defects discovered before adding further model complexity.

**Exit criteria:** a clean setup can audit data, train a model, run tests, generate actual test metrics and load a checkpoint in the app.

## Phase 1 — Classification research

- [ ] Benchmark CNN, ResNet18 and MobileNetV2 with the same split and preprocessing.
- [ ] Add ROC-AUC (one-vs-rest), PR-AUC, per-class sensitivity/specificity and confidence calibration.
- [ ] Add error analysis: false positives, false negatives and confidence bins.
- [ ] Add inference latency, checkpoint size and parameter count.
- [ ] Add experiment metadata and compare runs without overwriting prior results.
- [ ] Investigate perceptual near-duplicates and dataset provenance.

**Validation boundary:** the supplied dataset lacks adequate patient identifiers. Results must be described as image-level evaluation, not patient-level validation. External evaluation is only independent after source provenance and overlap have been checked.

## Phase 2 — U-Net segmentation research

- [x] Add a U-Net model architecture and shape tests.
- [x] Define paired image/mask manifest format.
- [ ] Obtain a legally usable, expert-annotated MRI segmentation dataset.
- [ ] Verify mask alignment, labels, provenance and patient identifiers.
- [ ] Create patient-disjoint train/validation/test partitions where identifiers permit.
- [ ] Implement segmentation training, Dice + BCE loss, Dice/IoU metrics and checkpointing.
- [ ] Validate visual overlays and failure cases with appropriate domain expertise.
- [ ] Connect a trained checkpoint to the segmentation dashboard page.

**Important:** the current four-class classification dataset has class labels, not pixel-level tumor masks. The U-Net architecture is present, but segmentation cannot be trained or evaluated from those class labels alone.

## Phase 3 — Explainability and trust

- [ ] Test Grad-CAM across all supported classifier architectures.
- [ ] Add Grad-CAM++ or another complementary explanation method.
- [ ] Add calibration analysis and uncertainty-aware messaging.
- [ ] Document explanation limitations and avoid interpreting heatmaps as tumor segmentation.

## Phase 4 — Application and reporting

- [ ] Add experiment history and model registry.
- [ ] Generate downloadable research reports with model/version metadata.
- [ ] Add input validation, file-size limits and safe temporary-file handling.
- [ ] Add clear dataset/model provenance and non-clinical-use notices.
- [ ] Test application behavior with missing, invalid and oversized files.

## Phase 5 — Public deployment

- [ ] Choose a hosting target based on model size, RAM, cold-start and inference requirements.
- [ ] Keep weights in release assets or external object storage rather than Git history.
- [ ] Configure dependency pinning and automated CI tests.
- [ ] Deploy a demo with a small, non-sensitive sample or user-provided test image.
- [ ] Add deployment monitoring and document limitations.

## Release gates

| Release | Required evidence |
|---|---|
| v0.1 — Reliable classifier | Reproducible CNN run, actual metrics, passing tests, app inference |
| v0.2 — Model benchmark | Comparable runs and error analysis for all classifier models |
| v0.3 — Segmentation prototype | Annotated data, trained U-Net, Dice/IoU and validated overlays |
| v0.4 — Research dashboard | Experiment tracking, reports and explainability |
| v1.0 — Public demo | CI, deployment, security checks and documented limitations |
