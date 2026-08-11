import torch

from landslide_sfda.engine import (
    configure_adaptation,
    set_adaptation_train_mode,
    train_steps,
)
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
    for mode in ("head", "bn", "decoder", "decoder-clean", "full"):
        model = UNet3DPaper(in_channels=11)
        counts[mode] = configure_adaptation(model, mode)
    assert counts["head"] < counts["bn"] < counts["decoder"] < counts["full"]
    assert counts["decoder-clean"] == counts["decoder"] == 1_826_881
    assert counts["full"] == 6_161_793
    assert counts["full"] == sum(
        parameter.numel() for parameter in UNet3DPaper(in_channels=11).parameters()
    )


def test_clean_decoder_freezes_encoder_training_state():
    model = UNet3DPaper(in_channels=11)
    configure_adaptation(model, "decoder-clean")
    set_adaptation_train_mode(model)
    for name in ("en3", "en4", "center_in", "center_out"):
        assert not getattr(model, name).training
    for name in ("dc4", "trans3", "dc3", "final"):
        assert getattr(model, name).training


def test_historical_decoder_retains_global_training_state():
    model = UNet3DPaper(in_channels=11)
    configure_adaptation(model, "decoder")
    set_adaptation_train_mode(model)
    assert model.en3.training
    assert model.center_out.training


def test_clean_decoder_keeps_encoder_weights_and_buffers_unchanged_after_step():
    model = UNet3DPaper(in_channels=11)
    configure_adaptation(model, "decoder-clean")
    frozen_before = {
        f"{module_name}.{state_name}": value.detach().clone()
        for module_name in ("en3", "en4", "center_in", "center_out")
        for state_name, value in getattr(model, module_name).state_dict().items()
    }
    final_before = model.final.weight.detach().clone()
    batch = {
        "x": torch.randn(1, 4, 11, 16, 16),
        "y": torch.randint(0, 2, (1, 1, 16, 16), dtype=torch.float32),
    }
    losses = train_steps(
        model,
        [batch],
        steps=1,
        learning_rate=1e-4,
        weight_decay=0.0,
        device=torch.device("cpu"),
        amp=False,
    )
    assert len(losses) == 1
    for key, before in frozen_before.items():
        module_name, state_name = key.split(".", 1)
        after = getattr(model, module_name).state_dict()[state_name]
        assert torch.equal(before, after), key
    assert not torch.equal(final_before, model.final.weight.detach())
