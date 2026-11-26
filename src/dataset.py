import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES=["glioma","meningioma","notumor","pituitary"]

class MRIDataset(Dataset):
    def __init__(self, manifest, partition, transform=None):
        df=pd.read_csv(manifest)
        self.df=df[(df.partition==partition) & df.usable].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No usable rows for partition: {partition}")
        self.transform=transform
    def __len__(self): return len(self.df)
    def __getitem__(self,index):
        row=self.df.iloc[index]
        with Image.open(row.path) as image:
            image=image.convert("RGB")
        if self.transform: image=self.transform(image)
        return image,int(row.label_id)
