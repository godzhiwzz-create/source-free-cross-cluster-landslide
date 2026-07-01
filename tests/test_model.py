import torch

from landslide_sfda.engine import configure_adaptation
from landslide_sfda.model import UNet3DPaper


def test_model_shape_and_parameter_count():
    model = UNet3DPaper(in_channels=11)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 4, 11, 16, 16))
    assert output.shape == (2, 1, 16, 16)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 6_150_000 < parameters < 6_170_000


def test_parameter_scopes_are_ordered():
    counts = {}
    for mode in ("head", "bn", "decoder", "full"):
        model = UNet3DPaper(in_channels=11)
        counts[mode] = configure_adaptation(model, mode)
    assert counts["head"] < counts["bn"] < counts["decoder"] < counts["full"]
    assert counts["full"] == sum(
        parameter.numel() for parameter in UNet3DPaper(in_channels=11).parameters()
    )
