from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

class MRISegmentationDataset(Dataset):
    REQUIRED={"image_path","mask_path","split"}
    def __init__(self,manifest,split,image_size=256):
        self.manifest=Path(manifest).resolve()
        frame=pd.read_csv(self.manifest)
        missing=self.REQUIRED-set(frame.columns)
        if missing: raise ValueError(f"Segmentation manifest missing columns: {sorted(missing)}")
        self.frame=frame[frame.split==split].reset_index(drop=True)
        if self.frame.empty: raise ValueError(f"No rows for split={split}")
        self.image_size=image_size
        base=self.manifest.parent
        for col in ("image_path","mask_path"):
            self.frame[col]=self.frame[col].map(lambda p:str((base/str(p)).resolve()) if not Path(str(p)).is_absolute() else str(Path(str(p)).resolve()))
        absent=[p for p in list(self.frame.image_path)+list(self.frame.mask_path) if not Path(p).is_file()]
        if absent: raise FileNotFoundError(f"Missing image or mask file: {absent[0]}")
        if "patient_id" in frame.columns and frame.patient_id.notna().any():
            split_patients=frame.dropna(subset=["patient_id"]).groupby("patient_id")["split"].nunique()
            if (split_patients>1).any(): raise ValueError("Patient leakage: a patient_id occurs in multiple splits.")
    def __len__(self): return len(self.frame)
    def __getitem__(self,index):
        row=self.frame.iloc[index]
        with Image.open(row.image_path) as im: image=im.convert("L")
        with Image.open(row.mask_path) as m: mask=m.convert("L")
        if image.size!=mask.size: raise ValueError(f"Image/mask dimensions differ: {row.image_path}")
        image=TF.resize(image,[self.image_size,self.image_size],interpolation=InterpolationMode.BILINEAR)
        mask=TF.resize(mask,[self.image_size,self.image_size],interpolation=InterpolationMode.NEAREST)
        image=TF.to_tensor(image)
        mask=(TF.to_tensor(mask)>0).float()
        return image,mask
