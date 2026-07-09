import numpy as np
import torch

from deeprx.matlab_bridge import (
    PaperFigure6Config,
    convert_official_batch_arrays,
    paper_dataset_frame_count,
    paper_dataset_split_frame_counts,
    pilot_count_to_dmrs_additional_position,
    sample_paper_dataset_parameters,
)


def test_paper_figure6_config_matches_paper_table_ii_and_figure_6a():
    config = PaperFigure6Config()

    assert config.carrier_frequency_hz == 4.0e9
    assert config.n_size_grid == 26
    assert config.subcarrier_spacing_khz == 15
    assert config.modulation == "16QAM"
    assert config.code_rate == 658 / 1024
    assert config.train_channels == ("CDL-B", "CDL-C", "CDL-D", "TDL-B", "TDL-C", "TDL-D")
    assert config.validation_channels == ("CDL-A", "CDL-E", "TDL-A", "TDL-E")
    assert config.figure6_sinr_points_db == (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0)
    assert config.dataset_ttis == 500_000
    assert config.ttis_per_frame == 10
    assert config.train_fraction == 0.6
    assert config.validation_samples_per_point == 500


def test_paper_dataset_split_uses_fixed_frame_counts():
    config = PaperFigure6Config()

    assert paper_dataset_frame_count(config) == 50_000
    assert paper_dataset_split_frame_counts(config) == (30_000, 20_000)


def test_validation_dataset_parameters_are_fixed_across_snr_points():
    config = PaperFigure6Config()

    low = sample_paper_dataset_parameters(config, split="validation", index=7, seed=2026, snr_db=0.0, pilot_count=2)
    high = sample_paper_dataset_parameters(config, split="validation", index=7, seed=2026, snr_db=21.0, pilot_count=2)
    one_pilot = sample_paper_dataset_parameters(config, split="validation", index=7, seed=2026, snr_db=21.0, pilot_count=1)

    assert low.snr_db == 0.0
    assert high.snr_db == 21.0
    assert low.channel_model == high.channel_model
    assert low.delay_spread_s == high.delay_spread_s
    assert low.max_doppler_shift_hz == high.max_doppler_shift_hz
    assert low.dmrs_configuration_type == high.dmrs_configuration_type
    assert one_pilot.dmrs_additional_position == 0
    assert high.dmrs_additional_position == 1
    assert one_pilot.channel_model == high.channel_model


def test_pilot_count_maps_to_official_dmrs_additional_position():
    assert pilot_count_to_dmrs_additional_position(1) == 0
    assert pilot_count_to_dmrs_additional_position(2) == 1


def test_convert_official_batch_arrays_reorders_matlab_axes_to_torch():
    f, s, c, bits, batch = 3, 2, 4, 2, 5
    x = np.arange(f * s * c * batch, dtype=np.float32).reshape(f, s, c, batch)
    targets = np.arange(f * s * bits * batch, dtype=np.float32).reshape(f, s, bits, batch) % 2
    data_mask = np.ones((f, s, 1, batch), dtype=np.float32)

    converted = convert_official_batch_arrays(x, targets, data_mask, "16QAM", max_bits_per_symbol=4)

    assert converted.inputs.shape == (batch, c, f, s)
    assert converted.target_bits.shape == (batch, bits, f, s)
    assert converted.data_mask.shape == (batch, 1, f, s)
    assert converted.bit_mask.shape == (1, 4, 1, 1)
    assert converted.inputs.dtype == torch.float32
    assert converted.bit_mask[:, :4].sum().item() == 4
    assert torch.equal(converted.inputs[0, :, :, :], torch.from_numpy(x[:, :, :, 0]).permute(2, 0, 1))
