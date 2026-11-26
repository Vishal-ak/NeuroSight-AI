import pandas as pd
import pytest
from PIL import Image
from src.segmentation_data import MRISegmentationDataset

def test_manifest_rejects_patient_leakage(tmp_path):
    image=tmp_path/"i.png"; mask=tmp_path/"m.png"
    Image.new("L",(16,16)).save(image); Image.new("L",(16,16)).save(mask)
    frame=pd.DataFrame([
      {"image_path":"i.png","mask_path":"m.png","split":"train","patient_id":"p1"},
      {"image_path":"i.png","mask_path":"m.png","split":"test","patient_id":"p1"}])
    manifest=tmp_path/"manifest.csv"; frame.to_csv(manifest,index=False)
    with pytest.raises(ValueError,match="Patient leakage"):
        MRISegmentationDataset(manifest,"train")
