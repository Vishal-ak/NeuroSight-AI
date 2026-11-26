from pathlib import Path
import pytest

from src.train import reject_output_images


def test_training_rejects_output_images_folder():
    with pytest.raises(ValueError, match="Output_images"):
        reject_output_images("data/Output_images")


def test_training_accepts_original_dataset_path():
    path = reject_output_images("data/raw/Brain_Tumor_MRI_Dataset")
    assert path.name == "Brain_Tumor_MRI_Dataset"
