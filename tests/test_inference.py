import torch
from src.models import build_model

def test_cnn_output_shape():
    model=build_model("cnn",4,False)
    assert tuple(model(torch.randn(2,3,224,224)).shape)==(2,4)
