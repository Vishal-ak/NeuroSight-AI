import torch.nn as nn
from torchvision import models

class SimpleCNN(nn.Module):
    def __init__(self,n=4):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.fc=nn.Linear(256,n)
    def forward(self,x): return self.fc(self.features(x).flatten(1))

def build_model(name="cnn",n=4,pretrained=True):
    if name=="cnn": return SimpleCNN(n)
    if name=="resnet18":
        model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc=nn.Linear(model.fc.in_features,n)
        return model
    if name=="mobilenet_v2":
        model=models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        model.classifier[1]=nn.Linear(model.classifier[1].in_features,n)
        return model
    raise ValueError(f"Unsupported model: {name}")
