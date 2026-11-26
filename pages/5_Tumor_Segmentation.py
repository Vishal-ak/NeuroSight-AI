from pathlib import Path
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from src.segmentation import UNet

st.set_page_config(page_title="Tumor Segmentation | NeuroSight AI",page_icon="🧩",layout="wide")
st.title("🧩 Tumor Segmentation")
st.caption("Experimental U-Net mask inference. A trained segmentation checkpoint and expert-annotated mask dataset are required.")
checkpoint=Path("models/segmentation/unet_best.pt")
if not checkpoint.exists():
    st.warning("No trained U-Net checkpoint is installed.")
    st.markdown("The current four-class classification dataset has no pixel-level masks. See docs/SEGMENTATION_DATA.md for the required paired image/mask data format.")
    st.code("python -m src.train_segmentation --manifest data/segmentation/manifest.csv --epochs 30")
    st.stop()
uploaded=st.file_uploader("Upload an MRI image",type=["png","jpg","jpeg","bmp"])
if uploaded:
    image=Image.open(uploaded).convert("L")
    ck=torch.load(checkpoint,map_location="cpu")
    size=int(ck.get("image_size",256))
    tensor=resize(torch.from_numpy(np.array(image)).unsqueeze(0),[size,size],interpolation=InterpolationMode.BILINEAR).float()/255.0
    model=UNet(in_channels=1,out_channels=1,base_channels=int(ck.get("base_channels",32)))
    model.load_state_dict(ck["state_dict"]); model.eval()
    with torch.no_grad():
        logits=model(tensor.unsqueeze(0))
        probability=torch.sigmoid(logits)[0,0].numpy()
    mask=Image.fromarray((probability>=float(ck.get("threshold",0.5))).astype("uint8")*255).resize(image.size,Image.Resampling.NEAREST)
    original=np.array(image)
    mask_array=np.array(mask)>0
    overlay=np.stack([original,original,original],axis=-1).astype(np.uint8)
    overlay[mask_array]=((0.35*overlay[mask_array])+(0.65*np.array([255,55,75]))).astype(np.uint8)
    a,b,c=st.columns(3)
    a.image(image,caption="Input MRI",use_container_width=True)
    b.image(mask,caption="Predicted binary mask",use_container_width=True)
    c.image(overlay,caption="Overlay (red = predicted mask)",use_container_width=True)
    st.metric("Predicted mask area",f"{100*mask_array.mean():.2f}%")
    st.warning("This is an experimental segmentation output, not a validated tumor boundary or clinical measurement.")
