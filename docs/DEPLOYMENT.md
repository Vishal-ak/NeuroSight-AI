# Public deployment notes

## Current deployment readiness

The Streamlit application can be deployed as a research interface, but a useful model demo requires a trained checkpoint. The repository intentionally excludes MRI datasets and trained weights.

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create a new app in Streamlit Community Cloud and select `app.py`.
3. Set Python version to 3.11 if prompted.
4. Deploy using `requirements.txt`.
5. Without a checkpoint, the prediction and segmentation pages should show setup guidance rather than fake results.

## Model artifacts

Do not commit large checkpoint files or raw medical images to Git. For a deployment with trained models, store weights in a controlled external artifact location or a GitHub Release if file size and licensing permit. Download artifacts during deployment using a secure, read-only mechanism. Never put private tokens in source code; use deployment secrets.

## Privacy and security checklist

- Do not store uploaded MRI images by default.
- Do not log image bytes, filenames containing patient identifiers, or personal health information.
- Keep upload size limits enabled.
- Validate file type and decode images safely.
- Do not use real patient-identifiable data in a public demo.
- Add a clear non-clinical-use notice and disclose model/data provenance.
- Public deployment is not evidence of clinical safety or regulatory approval.

## Current blocker

No trained classifier or U-Net checkpoint has been included. Train, evaluate and review model artifacts before advertising the application as a working prediction demo.
