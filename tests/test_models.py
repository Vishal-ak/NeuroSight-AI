import torch
from src.models import build_model

def test_cnn_output_shape():
    model=build_model("cnn",4,False)
    assert tuple(model(torch.randn(1,3,128,128)).shape)==(1,4)
