import torch
from src.segmentation import UNet,dice_loss,dice_score_from_logits

def test_unet_preserves_input_spatial_shape_for_even_dimensions():
    model=UNet(base_channels=8)
    output=model(torch.randn(2,1,128,128))
    assert tuple(output.shape)==(2,1,128,128)

def test_unet_handles_odd_dimensions():
    model=UNet(base_channels=8)
    output=model(torch.randn(1,1,129,131))
    assert tuple(output.shape)==(1,1,129,131)

def test_dice_metrics_are_finite():
    logits=torch.zeros(2,1,32,32)
    target=torch.zeros(2,1,32,32)
    assert torch.isfinite(dice_loss(logits,target))
    assert torch.isfinite(dice_score_from_logits(logits,target))
