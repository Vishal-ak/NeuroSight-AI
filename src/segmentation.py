import torch
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
    def forward(self,x): return self.layers(x)

class UNet(nn.Module):
    """U-Net for binary tumor-mask segmentation. Expects normalized 1-channel MRI tensors."""
    def __init__(self,in_channels=1,out_channels=1,base_channels=32):
        super().__init__()
        self.enc1=DoubleConv(in_channels,base_channels)
        self.enc2=DoubleConv(base_channels,base_channels*2)
        self.enc3=DoubleConv(base_channels*2,base_channels*4)
        self.enc4=DoubleConv(base_channels*4,base_channels*8)
        self.bottleneck=DoubleConv(base_channels*8,base_channels*16)
        self.pool=nn.MaxPool2d(2)
        self.up4=nn.ConvTranspose2d(base_channels*16,base_channels*8,2,2)
        self.dec4=DoubleConv(base_channels*16,base_channels*8)
        self.up3=nn.ConvTranspose2d(base_channels*8,base_channels*4,2,2)
        self.dec3=DoubleConv(base_channels*8,base_channels*4)
        self.up2=nn.ConvTranspose2d(base_channels*4,base_channels*2,2,2)
        self.dec2=DoubleConv(base_channels*4,base_channels*2)
        self.up1=nn.ConvTranspose2d(base_channels*2,base_channels,2,2)
        self.dec1=DoubleConv(base_channels*2,base_channels)
        self.head=nn.Conv2d(base_channels,out_channels,1)

    @staticmethod
    def _match(x,skip):
        if x.shape[-2:]!=skip.shape[-2:]:
            x=F.interpolate(x,size=skip.shape[-2:],mode="bilinear",align_corners=False)
        return x

    def forward(self,x):
        e1=self.enc1(x)
        e2=self.enc2(self.pool(e1))
        e3=self.enc3(self.pool(e2))
        e4=self.enc4(self.pool(e3))
        b=self.bottleneck(self.pool(e4))
        d4=self.dec4(torch.cat([self._match(self.up4(b),e4),e4],dim=1))
        d3=self.dec3(torch.cat([self._match(self.up3(d4),e3),e3],dim=1))
        d2=self.dec2(torch.cat([self._match(self.up2(d3),e2),e2],dim=1))
        d1=self.dec1(torch.cat([self._match(self.up1(d2),e1),e1],dim=1))
        return self.head(d1)

def dice_score_from_logits(logits,target,threshold=0.5,eps=1e-7):
    prediction=(torch.sigmoid(logits)>=threshold).float()
    dims=(1,2,3)
    intersection=(prediction*target).sum(dims)
    denominator=prediction.sum(dims)+target.sum(dims)
    return ((2*intersection+eps)/(denominator+eps)).mean()

def dice_loss(logits,target,eps=1e-7):
    probabilities=torch.sigmoid(logits)
    dims=(1,2,3)
    intersection=(probabilities*target).sum(dims)
    denominator=probabilities.sum(dims)+target.sum(dims)
    return 1-((2*intersection+eps)/(denominator+eps)).mean()
