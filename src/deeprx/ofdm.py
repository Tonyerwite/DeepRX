import math
from typing import Tuple

import torch


class OFDMTransmitter:
    def __init__(self, n_subcarriers: int = 312, n_fft: int = 512, cp_length: int = 36, n_symbols: int = 14):
        self.n_subcarriers = n_subcarriers
        self.n_fft = n_fft
        self.cp_length = cp_length
        self.n_symbols = n_symbols

    def build_resource_grid(
        self,
        data_symbols: torch.Tensor,
        pilot_symbols: torch.Tensor,
        pilot_mask: torch.Tensor,
        data_bits: torch.Tensor,
        max_bits: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = data_symbols.shape[0]
        device = data_symbols.device
        grid = pilot_symbols.clone().to(device)
        data_mask = (1.0 - pilot_mask.to(device))[0, 0].bool()
        n_data = int(data_mask.sum().item())
        if data_symbols.shape[1] != n_data:
            raise ValueError(f"Expected {n_data} data symbols, got {data_symbols.shape[1]}")

        target = torch.zeros(batch, max_bits, self.n_symbols, self.n_subcarriers, device=device)
        for batch_idx in range(batch):
            grid[batch_idx, 0, data_mask] = data_symbols[batch_idx]
            bits = data_bits[batch_idx]
            target[batch_idx, : bits.shape[-1], data_mask] = bits.T
        return grid, target

    def modulate(self, grid: torch.Tensor) -> torch.Tensor:
        batch = grid.shape[0]
        device = grid.device
        fft_grid = torch.zeros(batch, self.n_symbols, self.n_fft, dtype=torch.cfloat, device=device)
        start = (self.n_fft - self.n_subcarriers) // 2
        fft_grid[:, :, start : start + self.n_subcarriers] = grid[:, 0]
        time_symbols = torch.fft.ifft(torch.fft.ifftshift(fft_grid, dim=-1), dim=-1)
        cp = time_symbols[:, :, -self.cp_length :]
        return torch.cat([cp, time_symbols], dim=-1).reshape(batch, -1)


class OFDMReceiver:
    def __init__(self, n_subcarriers: int = 312, n_fft: int = 512, cp_length: int = 36, n_symbols: int = 14):
        self.n_subcarriers = n_subcarriers
        self.n_fft = n_fft
        self.cp_length = cp_length
        self.n_symbols = n_symbols

    def demodulate(self, rx_waveform: torch.Tensor, n_rx: int = 1) -> torch.Tensor:
        if rx_waveform.dim() == 2:
            rx_waveform = rx_waveform.unsqueeze(1)
        batch = rx_waveform.shape[0]
        symbol_len = self.n_fft + self.cp_length
        start_sc = (self.n_fft - self.n_subcarriers) // 2
        grid = torch.zeros(batch, n_rx, self.n_symbols, self.n_subcarriers, dtype=torch.cfloat, device=rx_waveform.device)
        for symbol_idx in range(self.n_symbols):
            start = symbol_idx * symbol_len + self.cp_length
            end = start + self.n_fft
            symbol = rx_waveform[:, :, start:end]
            freq = torch.fft.fftshift(torch.fft.fft(symbol, dim=-1), dim=-1)
            grid[:, :, symbol_idx] = freq[:, :, start_sc : start_sc + self.n_subcarriers]
        return grid


class ChannelModel:
    """Small, deterministic-friendly approximation of 3GPP TDL/CDL fading channels."""

    profiles = {
        "TDL-A": ([0, 1, 2, 3, 5, 7], [0, -3.6, -7.2, -10.0, -14.0, -18.0], False),
        "TDL-B": ([0, 1, 2, 4, 6, 9], [0, -2.2, -4.0, -6.0, -9.0, -13.0], False),
        "TDL-C": ([0, 1, 3, 5, 8, 12], [0, -4.4, -6.2, -8.2, -11.0, -15.0], False),
        "TDL-D": ([0, 1, 2, 4, 7, 10], [-0.2, -8.0, -10.0, -12.0, -16.0, -20.0], True),
        "TDL-E": ([0, 1, 2, 5, 9, 14], [-0.03, -1.2, -2.1, -5.0, -8.0, -12.0], True),
        "CDL-A": ([0, 1, 2, 4, 6, 8], [0, -3.0, -5.0, -8.0, -12.0, -16.0], False),
        "CDL-B": ([0, 1, 3, 4, 7, 10], [0, -2.0, -3.5, -7.0, -11.0, -15.0], False),
        "CDL-C": ([0, 2, 3, 6, 8, 12], [0, -2.5, -4.5, -7.5, -10.5, -14.5], False),
        "CDL-D": ([0, 1, 2, 4, 6, 9], [-0.2, -8.2, -10.3, -12.1, -15.0, -19.0], True),
        "CDL-E": ([0, 1, 3, 5, 9, 13], [-0.03, -1.0, -2.0, -5.0, -8.0, -13.0], True),
        "SYNTHETIC": ([0, 1, 2, 3, 4, 5, 6], [0, -2, -4, -6, -8, -10, -12], False),
    }

    def __init__(self, profile: str = "TDL-B", max_doppler_hz: float = 100.0, device: str = "cpu"):
        self.profile = profile.upper().replace("_", "-")
        if self.profile not in self.profiles:
            raise ValueError(f"Unsupported channel profile: {profile}")
        delays, powers_db, is_los = self.profiles[self.profile]
        self.delays = torch.tensor(delays, dtype=torch.long, device=device)
        powers = 10 ** (torch.tensor(powers_db, dtype=torch.float32, device=device) / 10.0)
        self.powers = powers / powers.sum()
        self.max_doppler_hz = max_doppler_hz
        self.is_los = is_los
        self.device = torch.device(device)

    def generate_taps(self, batch: int, n_symbols: int = 14, symbol_duration_s: float = 1e-3 / 14) -> torch.Tensor:
        n_taps = len(self.delays)
        rho = math.exp(-2.0 * math.pi * self.max_doppler_hz * symbol_duration_s)
        rho = min(max(rho, 0.0), 0.9999)
        taps = torch.zeros(batch, n_symbols, n_taps, dtype=torch.cfloat, device=self.device)
        prev = None
        for symbol_idx in range(n_symbols):
            noise = torch.complex(torch.randn(batch, n_taps, device=self.device), torch.randn(batch, n_taps, device=self.device))
            noise = noise * torch.sqrt(self.powers / 2.0)
            if prev is None:
                current = noise
            else:
                current = rho * prev + math.sqrt(1.0 - rho * rho) * noise
            if self.is_los:
                current[:, 0] = current[:, 0] + torch.sqrt(self.powers[0] * 3.0)
            taps[:, symbol_idx] = current
            prev = current
        return taps

    def apply(self, waveform: torch.Tensor, taps: torch.Tensor, n_fft: int, cp_length: int) -> torch.Tensor:
        batch, signal_len = waveform.shape
        out = torch.zeros_like(waveform)
        symbol_len = n_fft + cp_length
        for symbol_idx in range(taps.shape[1]):
            start = symbol_idx * symbol_len
            end = min(start + symbol_len, signal_len)
            segment = waveform[:, start:end]
            for tap_idx, delay in enumerate(self.delays.tolist()):
                coeff = taps[:, symbol_idx, tap_idx].unsqueeze(-1)
                if delay == 0:
                    out[:, start:end] += coeff * segment
                elif delay < segment.shape[-1]:
                    out[:, start + delay : end] += coeff * segment[:, : segment.shape[-1] - delay]
        return out


def add_awgn(signal: torch.Tensor, snr_db: float, signal_power: float = None) -> torch.Tensor:
    if signal_power is None:
        signal_power = float((signal.abs() ** 2).mean().detach().cpu().item())
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    std = math.sqrt(noise_power / 2.0)
    noise = torch.complex(torch.randn_like(signal.real), torch.randn_like(signal.imag)) * std
    return signal + noise
