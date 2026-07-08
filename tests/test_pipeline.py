from pathlib import Path

import torch

from deeprx.data import DeepRxDataset
from deeprx.experiments import evaluate_checkpoint_snr_grid, run_quick_reproduction
from deeprx.model import DeepRx


def test_dataset_sample_matches_mathworks_default_tensor_sizes():
    dataset = DeepRxDataset(
        n_samples=1,
        n_rx_antennas=2,
        n_subcarriers=312,
        n_fft=512,
        cp_length=36,
        n_symbols=14,
        modulation="16QAM",
        snr_range=(12.0, 12.0),
        doppler_range=(0.0, 0.0),
        channel_profiles=("TDL-B",),
        pilot_configs=("2_pilots_left",),
    )
    sample = dataset[0]

    assert sample.inputs.shape == (10, 14, 312)
    assert sample.target_bits.shape == (8, 14, 312)
    assert sample.data_mask.shape == (1, 14, 312)
    assert sample.bit_mask.shape == (8, 1, 1)


def test_dataset_can_emit_mathworks_16qam_four_bit_targets():
    dataset = DeepRxDataset(
        n_samples=1,
        n_subcarriers=312,
        n_fft=512,
        cp_length=36,
        modulation="16QAM",
        snr_range=(12.0, 12.0),
        channel_profiles=("TDL-B",),
        pilot_configs=("2_pilots_left",),
        max_bits_per_symbol=4,
    )
    sample = dataset[0]

    assert sample.target_bits.shape == (4, 14, 312)
    assert sample.bit_mask.shape == (4, 1, 1)


def test_one_optimizer_step_updates_deeprx_parameters():
    model = DeepRx(n_rx_antennas=2, max_bits_per_symbol=8)
    dataset = DeepRxDataset(
        n_samples=1,
        n_subcarriers=72,
        n_fft=128,
        cp_length=16,
        modulation="QPSK",
        snr_range=(15.0, 15.0),
        channel_profiles=("TDL-B",),
        pilot_configs=("2_pilots_left",),
    )
    sample = dataset[0]
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    loss_value = model.train_step(
        sample.inputs.unsqueeze(0),
        sample.target_bits.unsqueeze(0),
        sample.data_mask,
        sample.bit_mask,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
    )

    changed = any(not torch.allclose(before[name], param) for name, param in model.named_parameters())
    assert loss_value > 0
    assert changed


def test_quick_reproduction_writes_metrics_and_figures(tmp_path: Path):
    results = run_quick_reproduction(output_dir=tmp_path, train_steps=1, eval_batches=1, batch_size=1)

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "ber_vs_snr.png").exists()
    assert (tmp_path / "ber_vs_doppler.png").exists()
    assert set(results) >= {"snr_db", "deeprx_ber", "lmmse_ber", "doppler_hz"}


def test_quick_reproduction_defaults_to_two_db_snr_grid_through_12_db(tmp_path: Path):
    results = run_quick_reproduction(output_dir=tmp_path, train_steps=0, eval_batches=1, batch_size=1)

    assert results["snr_db"] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


def test_evaluate_checkpoint_snr_grid_writes_metrics_and_figure(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": DeepRx(max_bits_per_symbol=4).state_dict()}, checkpoint)

    results = evaluate_checkpoint_snr_grid(
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "eval",
        snr_points=[0.0, 2.0],
        eval_samples=1,
        n_subcarriers=72,
        n_fft=128,
        cp_length=16,
        max_bits_per_symbol=4,
    )

    assert results["snr_db"] == [0.0, 2.0]
    assert (tmp_path / "eval" / "checkpoint_ber_vs_snr.png").exists()
    assert (tmp_path / "eval" / "checkpoint_metrics.json").exists()
