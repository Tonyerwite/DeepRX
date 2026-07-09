from pathlib import Path

import torch

from deeprx.model import (
    DeepRx,
    DeepRxLoss,
    compute_ber,
    create_bit_mask,
)


ROOT = Path(__file__).resolve().parents[1]
MATHWORKS_PYTORCH_CHECKPOINT = (
    ROOT / "official" / "deeprx_30k.pth"
)


def test_deeprx_model_shape_parameter_count_and_fully_convolutional_width():
    model = DeepRx(n_rx_antennas=2, max_bits_per_symbol=8)
    z = torch.randn(2, 10, 14, 312)
    out = model(z)

    assert out.shape == (2, 8, 14, 312)
    assert 1_200_000 <= model.count_parameters() <= 1_260_000

    narrow = torch.randn(1, 10, 14, 96)
    assert model(narrow).shape == (1, 8, 14, 96)


def test_deeprx_four_bit_model_matches_mathworks_parameter_count():
    model = DeepRx(n_rx_antennas=2, max_bits_per_symbol=4)

    assert model.count_parameters() == 1_232_516


def test_deeprx_loads_mathworks_pytorch_checkpoint_when_available():
    if not MATHWORKS_PYTORCH_CHECKPOINT.exists():
        return
    model = DeepRx(n_rx_antennas=2, max_bits_per_symbol=4)
    state = torch.load(MATHWORKS_PYTORCH_CHECKPOINT, map_location="cpu", weights_only=True)

    model.load_state_dict(state)
    out = model(torch.randn(1, 10, 312, 14))

    assert out.shape == (1, 4, 312, 14)


def test_loss_and_ber_use_positive_logits_for_bit_one():
    target = torch.randint(0, 2, (2, 8, 14, 32)).float()
    data_mask = torch.ones(1, 1, 14, 32)
    bit_mask = create_bit_mask("16QAM")
    perfect_logits = (target * 2.0 - 1.0) * 8.0

    assert compute_ber(perfect_logits, target, data_mask, bit_mask) == 0.0
    assert compute_ber(-perfect_logits, target, data_mask, bit_mask) == 1.0

    loss = DeepRxLoss()(perfect_logits, target, data_mask, bit_mask)
    assert loss.item() < 0.001
