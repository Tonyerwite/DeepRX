import torch

from deeprx.model import (
    DeepRx,
    DeepRxLoss,
    build_deeprx_input,
    compute_ber,
    create_bit_mask,
    create_pilot_mask,
    generate_qpsk_pilots,
)


def test_paper_pilot_masks_match_one_prb_pattern_counts():
    one = create_pilot_mask(14, 312, "1_pilot_left")
    two = create_pilot_mask(14, 312, "2_pilots_left")

    assert one.shape == (1, 1, 14, 312)
    assert two.shape == (1, 1, 14, 312)
    assert int(one.sum().item()) == 156
    assert int(two.sum().item()) == 312


def test_deeprx_input_stacks_rx_pilots_and_raw_channel_estimates():
    batch, nr, symbols, subcarriers = 2, 2, 14, 312
    pilot_mask = create_pilot_mask(symbols, subcarriers, "1_pilot_left")
    rx_grid = torch.randn(batch, nr, symbols, subcarriers, dtype=torch.cfloat)
    tx_pilots = generate_qpsk_pilots(batch, symbols, subcarriers, pilot_mask)

    z = build_deeprx_input(rx_grid, tx_pilots)
    nc = 2 * nr + 1

    assert z.shape == (batch, 2 * nc, symbols, subcarriers)
    raw_real = z[:, nr + 1 : nr + 1 + nr]
    raw_imag = z[:, nc + nr + 1 : nc + nr + 1 + nr]
    raw_complex = torch.complex(raw_real, raw_imag)
    expected_raw = rx_grid * torch.conj(tx_pilots.expand(-1, nr, -1, -1))
    assert torch.allclose(raw_complex, expected_raw)


def test_deeprx_model_shape_parameter_count_and_fully_convolutional_width():
    model = DeepRx(n_rx_antennas=2, max_bits_per_symbol=8)
    z = torch.randn(2, 10, 14, 312)
    out = model(z)

    assert out.shape == (2, 8, 14, 312)
    assert 1_200_000 <= model.count_parameters() <= 1_260_000

    narrow = torch.randn(1, 10, 14, 96)
    assert model(narrow).shape == (1, 8, 14, 96)


def test_loss_and_ber_use_positive_logits_for_bit_one():
    target = torch.randint(0, 2, (2, 8, 14, 32)).float()
    data_mask = torch.ones(1, 1, 14, 32)
    bit_mask = create_bit_mask("16QAM")
    perfect_logits = (target * 2.0 - 1.0) * 8.0

    assert compute_ber(perfect_logits, target, data_mask, bit_mask) == 0.0
    assert compute_ber(-perfect_logits, target, data_mask, bit_mask) == 1.0

    loss = DeepRxLoss()(perfect_logits, target, data_mask, bit_mask)
    assert loss.item() < 0.001
