import random
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from deeprx.config import PAPER_CONFIG
from deeprx.model import build_deeprx_input, create_bit_mask, create_pilot_mask, generate_qpsk_pilots
from deeprx.ofdm import ChannelModel, OFDMReceiver, OFDMTransmitter, add_awgn
from deeprx.qam import QAMModem


@dataclass
class DeepRxSample:
    inputs: torch.Tensor
    target_bits: torch.Tensor
    data_mask: torch.Tensor
    bit_mask: torch.Tensor
    rx_grid: torch.Tensor
    tx_pilots: torch.Tensor
    pilot_mask: torch.Tensor
    snr_db: float
    doppler_hz: float
    channel_profile: str


class DeepRxDataset(torch.utils.data.Dataset):
    """Online data generator matching the DeepRx supervised-learning setup."""

    def __init__(
        self,
        n_samples: int = 30_000,
        n_rx_antennas: int = PAPER_CONFIG.n_rx_antennas,
        n_subcarriers: int = PAPER_CONFIG.n_subcarriers,
        n_fft: int = PAPER_CONFIG.n_fft,
        cp_length: int = PAPER_CONFIG.cp_length,
        n_symbols: int = PAPER_CONFIG.n_symbols,
        modulation: str = PAPER_CONFIG.modulation,
        snr_range: Tuple[float, float] = PAPER_CONFIG.snr_range_db,
        doppler_range: Tuple[float, float] = PAPER_CONFIG.doppler_range_hz,
        channel_profiles: Sequence[str] = PAPER_CONFIG.train_channels,
        pilot_configs: Sequence[str] = PAPER_CONFIG.pilot_configs,
        max_bits_per_symbol: int = 8,
        device: str = "cpu",
    ):
        self.n_samples = n_samples
        self.n_rx_antennas = n_rx_antennas
        self.n_subcarriers = n_subcarriers
        self.n_fft = n_fft
        self.cp_length = cp_length
        self.n_symbols = n_symbols
        self.modulation = modulation.upper()
        self.snr_range = snr_range
        self.doppler_range = doppler_range
        self.channel_profiles = tuple(channel_profiles)
        self.pilot_configs = tuple(pilot_configs)
        self.max_bits_per_symbol = max_bits_per_symbol
        self.device = torch.device(device)
        self.modem = QAMModem(self.modulation, device=device)
        self.tx = OFDMTransmitter(n_subcarriers, n_fft, cp_length, n_symbols)
        self.rx = OFDMReceiver(n_subcarriers, n_fft, cp_length, n_symbols)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> DeepRxSample:
        del index
        snr_db = random.uniform(*self.snr_range)
        doppler_hz = random.uniform(*self.doppler_range)
        profile = random.choice(self.channel_profiles)
        pilot_config = random.choice(self.pilot_configs)

        pilot_mask = create_pilot_mask(self.n_symbols, self.n_subcarriers, pilot_config, self.device)
        data_mask = 1.0 - pilot_mask
        bit_mask = create_bit_mask(self.modulation, max_bits=self.max_bits_per_symbol, device=self.device).squeeze(0)
        n_data = int(data_mask.sum().item())

        bits = torch.randint(0, 2, (1, n_data, self.modem.bits_per_symbol), device=self.device).float()
        symbols = self.modem.modulate(bits.reshape(-1, self.modem.bits_per_symbol)).reshape(1, n_data)
        pilots = generate_qpsk_pilots(1, self.n_symbols, self.n_subcarriers, pilot_mask, self.device)
        grid, target_bits = self.tx.build_resource_grid(
            symbols,
            pilots,
            pilot_mask,
            bits,
            max_bits=self.max_bits_per_symbol,
        )
        waveform = self.tx.modulate(grid)
        signal_power = float((waveform.abs() ** 2).mean().detach().cpu().item())

        rx_antennas = []
        for _ in range(self.n_rx_antennas):
            channel = ChannelModel(profile, max_doppler_hz=doppler_hz, device=self.device)
            taps = channel.generate_taps(1, self.n_symbols)
            faded = channel.apply(waveform, taps, self.n_fft, self.cp_length)
            rx_antennas.append(add_awgn(faded, snr_db, signal_power).squeeze(0))
        rx_waveform = torch.stack(rx_antennas, dim=0).unsqueeze(0)
        rx_grid = self.rx.demodulate(rx_waveform, self.n_rx_antennas)
        inputs = build_deeprx_input(rx_grid, pilots)

        return DeepRxSample(
            inputs=inputs.squeeze(0),
            target_bits=target_bits.squeeze(0),
            data_mask=data_mask.squeeze(0),
            bit_mask=bit_mask,
            rx_grid=rx_grid.squeeze(0),
            tx_pilots=pilots.squeeze(0),
            pilot_mask=pilot_mask.squeeze(0),
            snr_db=snr_db,
            doppler_hz=doppler_hz,
            channel_profile=profile,
        )


def collate_samples(samples):
    return {
        "inputs": torch.stack([sample.inputs for sample in samples]),
        "target_bits": torch.stack([sample.target_bits for sample in samples]),
        "data_mask": torch.stack([sample.data_mask for sample in samples]),
        "bit_mask": torch.stack([sample.bit_mask for sample in samples]),
        "rx_grid": torch.stack([sample.rx_grid for sample in samples]),
        "tx_pilots": torch.stack([sample.tx_pilots for sample in samples]),
        "pilot_mask": torch.stack([sample.pilot_mask for sample in samples]),
        "snr_db": torch.tensor([sample.snr_db for sample in samples]),
        "doppler_hz": torch.tensor([sample.doppler_hz for sample in samples]),
    }
