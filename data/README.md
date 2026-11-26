# Dataset setup

Extract the supplied four-class collection into data/raw/Brain_Tumor_MRI_Dataset/ with Training/{glioma,meningioma,notumor,pituitary} and Testing/{glioma,meningioma,notumor,pituitary} directories.

Do not commit MRI images. Run the audit script before training. Exact SHA-256 deduplication does not detect perceptual near-duplicates or establish patient-level independence. Check dataset licensing and provenance before redistribution.
