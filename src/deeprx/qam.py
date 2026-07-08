import math
from functools import lru_cache

import torch


_BITS_PER_SYMBOL = {
    "QPSK": 2,
    "16QAM": 4,
    "64QAM": 6,
    "256QAM": 8,
}


def bits_per_symbol(modulation: str) -> int:
    try:
        return _BITS_PER_SYMBOL[modulation.upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported modulation: {modulation}") from exc


def _gray_to_binary(value: int) -> int:
    result = value
    while value > 0:
        value >>= 1
        result ^= value
    return result


@lru_cache(maxsize=16)
def _constellation_cpu(modulation: str):
    modulation = modulation.upper()
    bps = bits_per_symbol(modulation)
    bits_dim = bps // 2
    dim = 2**bits_dim
    levels = torch.arange(dim, dtype=torch.float32) * 2.0 - (dim - 1)

    level_by_label = torch.empty(dim, dtype=torch.float32)
    for label in range(dim):
        level_by_label[label] = levels[_gray_to_binary(label)]

    points = []
    labels = []
    for i_label in range(dim):
        for q_label in range(dim):
            point = torch.complex(level_by_label[i_label], level_by_label[q_label])
            points.append(point)
            labels.append(i_label * dim + q_label)

    constellation = torch.stack(points)
    constellation = constellation / torch.sqrt((constellation.abs() ** 2).mean())
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return constellation, label_tensor, bps


class QAMModem:
    """Gray-labelled square QAM modem with unit-average-power constellations."""

    def __init__(self, modulation: str = "16QAM", device: str = "cpu"):
        self.modulation = modulation.upper()
        self.device = torch.device(device)
        constellation, labels, bps = _constellation_cpu(self.modulation)
        self.constellation = constellation.to(self.device)
        self.labels = labels.to(self.device)
        self.bits_per_symbol = bps

        bit_rows = []
        for label in self.labels.tolist():
            bits = [(label >> shift) & 1 for shift in range(bps - 1, -1, -1)]
            bit_rows.append(bits)
        self.bit_labels = torch.tensor(bit_rows, dtype=torch.float32, device=self.device)

    def to(self, device: str):
        return QAMModem(self.modulation, device=device)

    def modulate(self, bits: torch.Tensor) -> torch.Tensor:
        if bits.shape[-1] != self.bits_per_symbol:
            raise ValueError(f"Expected {self.bits_per_symbol} bits per symbol, got {bits.shape[-1]}")
        powers = (2 ** torch.arange(self.bits_per_symbol - 1, -1, -1, device=bits.device)).float()
        labels = (bits.float() * powers).sum(dim=-1).long()
        return self.constellation.to(bits.device)[labels]

    def hard_demodulate(self, symbols: torch.Tensor) -> torch.Tensor:
        constellation = self.constellation.to(symbols.device)
        bit_labels = self.bit_labels.to(symbols.device)
        distances = (symbols.reshape(-1, 1) - constellation.reshape(1, -1)).abs()
        nearest = distances.argmin(dim=-1)
        bits = bit_labels[nearest]
        return bits.reshape(*symbols.shape, self.bits_per_symbol)

    def random_bits_and_symbols(self, n_symbols: int, device: str = "cpu"):
        bits = torch.randint(0, 2, (n_symbols, self.bits_per_symbol), device=device).float()
        return bits, self.modulate(bits)


def max_log_llrs(symbols: torch.Tensor, noise_scale: torch.Tensor, modem: QAMModem, b_max: int = 8) -> torch.Tensor:
    constellation = modem.constellation.to(symbols.device)
    bit_labels = modem.bit_labels.to(symbols.device)
    distances = (symbols.unsqueeze(-1) - constellation.view(1, 1, 1, -1)).abs() ** 2
    llrs = torch.zeros(symbols.shape[0], b_max, symbols.shape[1], symbols.shape[2], device=symbols.device)

    for bit_idx in range(modem.bits_per_symbol):
        zero = bit_labels[:, bit_idx] == 0
        one = bit_labels[:, bit_idx] == 1
        min_zero = distances[..., zero].min(dim=-1).values
        min_one = distances[..., one].min(dim=-1).values
        llrs[:, bit_idx] = noise_scale * (min_zero - min_one)
    return llrs
