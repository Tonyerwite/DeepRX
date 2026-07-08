import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


MODULATION_BITS = {
    "QPSK": 2,
    "16QAM": 4,
    "64QAM": 6,
    "256QAM": 8,
}


def create_bit_mask(modulation: str, max_bits: int = 8, device: str = "cpu") -> torch.Tensor:
    modulation = modulation.upper()
    if modulation not in MODULATION_BITS:
        raise ValueError(f"Unsupported modulation: {modulation}")
    if max_bits < MODULATION_BITS[modulation]:
        raise ValueError(f"max_bits={max_bits} cannot represent {modulation}")
    mask = torch.zeros(1, max_bits, 1, 1, device=device)
    mask[:, : MODULATION_BITS[modulation]] = 1.0
    return mask


def create_pilot_mask(
    n_symbols: int = 14,
    n_subcarriers: int = 312,
    config: str = "2_pilots_left",
    device: str = "cpu",
) -> torch.Tensor:
    aliases = {
        "1_pilot_A": "1_pilot_left",
        "1_pilot_B": "1_pilot_right",
        "2_pilots_A": "2_pilots_left",
        "2_pilots_B": "2_pilots_right",
    }
    config = aliases.get(config, config)
    mask = torch.zeros(1, 1, n_symbols, n_subcarriers, device=device)
    if config == "1_pilot_left":
        mask[0, 0, 2, 0::2] = 1.0
    elif config == "1_pilot_right":
        mask[0, 0, 2, 1::2] = 1.0
    elif config == "2_pilots_left":
        mask[0, 0, 2, 0::2] = 1.0
        mask[0, 0, 11, 1::2] = 1.0
    elif config == "2_pilots_right":
        mask[0, 0, 2, 1::2] = 1.0
        mask[0, 0, 11, 0::2] = 1.0
    else:
        raise ValueError(f"Unknown pilot config: {config}")
    return mask


def generate_qpsk_pilots(
    batch_size: int,
    n_symbols: int,
    n_subcarriers: int,
    pilot_mask: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    real = 2.0 * torch.randint(0, 2, (batch_size, 1, n_symbols, n_subcarriers), device=device).float() - 1.0
    imag = 2.0 * torch.randint(0, 2, (batch_size, 1, n_symbols, n_subcarriers), device=device).float() - 1.0
    pilots = torch.complex(real, imag) / math.sqrt(2.0)
    return pilots * pilot_mask.to(device)


def build_deeprx_input(rx_grid: torch.Tensor, tx_pilots: torch.Tensor) -> torch.Tensor:
    if rx_grid.dtype not in (torch.complex64, torch.complex128):
        raise TypeError("rx_grid must be complex")
    n_rx = rx_grid.shape[1]
    pilots_expanded = tx_pilots.expand(-1, n_rx, -1, -1)
    raw_channel_estimate = rx_grid * torch.conj(pilots_expanded)
    z_complex = torch.cat([rx_grid, tx_pilots, raw_channel_estimate], dim=1)
    return torch.cat([z_complex.real, z_complex.imag], dim=1).float()


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        dilation: Tuple[int, int] = (1, 1),
        depth_multiplier: int = 2,
    ):
        super().__init__()
        mid_channels = in_channels * depth_multiplier
        padding = (
            dilation[0] * (kernel_size[0] - 1) // 2,
            dilation[1] * (kernel_size[1] - 1) // 2,
        )
        self.depthwise = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class PreactivationResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: Tuple[int, int],
        depth_multiplier: int = 2,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = DepthwiseSeparableConv2d(in_channels, out_channels, dilation=dilation, depth_multiplier=depth_multiplier)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv2d(out_channels, out_channels, dilation=dilation, depth_multiplier=depth_multiplier)
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.projection is None else self.projection(x)
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + identity


class DeepRx(nn.Module):
    """Fully convolutional DeepRx network from Honkala et al. Table I."""

    block_configs = (
        (64, (1, 1)),
        (64, (1, 1)),
        (128, (2, 3)),
        (128, (2, 3)),
        (256, (2, 3)),
        (256, (3, 6)),
        (256, (2, 3)),
        (128, (2, 3)),
        (128, (2, 3)),
        (64, (1, 1)),
        (64, (1, 1)),
    )

    def __init__(self, n_rx_antennas: int = 2, max_bits_per_symbol: int = 8, depth_multiplier: int = 2):
        super().__init__()
        self.n_rx_antennas = n_rx_antennas
        self.n_rx = n_rx_antennas
        self.max_bits_per_symbol = max_bits_per_symbol
        n_input_channels = 2 * (2 * n_rx_antennas + 1)

        self.conv_in = nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1, bias=False)
        blocks = []
        in_channels = 64
        for out_channels, dilation in self.block_configs:
            blocks.append(PreactivationResNetBlock(in_channels, out_channels, dilation, depth_multiplier))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.bn_out = nn.BatchNorm2d(in_channels)
        self.conv_out = nn.Conv2d(in_channels, max_bits_per_symbol, kernel_size=3, padding=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(z)
        for block in self.blocks:
            out = block(out)
        return self.conv_out(F.relu(self.bn_out(out)))

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def train_step(
        self,
        inputs: torch.Tensor,
        target_bits: torch.Tensor,
        data_mask: torch.Tensor,
        bit_mask: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.train()
        optimizer.zero_grad(set_to_none=True)
        logits = self(inputs)
        loss = DeepRxLoss()(logits, target_bits, data_mask, bit_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        return float(loss.detach().cpu().item())


class DeepRxLoss(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        target_bits: torch.Tensor,
        data_mask: torch.Tensor,
        bit_mask: torch.Tensor,
    ) -> torch.Tensor:
        data_mask = _normalize_data_mask(data_mask, logits.device)
        bit_mask = _normalize_bit_mask(bit_mask, logits.device)
        full_mask = (data_mask * bit_mask).expand_as(logits)
        losses = F.binary_cross_entropy_with_logits(logits, target_bits.to(logits.device).float(), reduction="none")
        return (losses * full_mask).sum() / full_mask.sum().clamp_min(1.0)


def compute_ber(logits: torch.Tensor, target_bits: torch.Tensor, data_mask: torch.Tensor, bit_mask: torch.Tensor) -> float:
    data_mask = _normalize_data_mask(data_mask, logits.device)
    bit_mask = _normalize_bit_mask(bit_mask, logits.device)
    full_mask = (data_mask * bit_mask).expand_as(logits)
    decisions = (logits > 0).float()
    errors = ((decisions != target_bits.to(logits.device).float()).float() * full_mask).sum()
    total = full_mask.sum().clamp_min(1.0)
    return float((errors / total).detach().cpu().item())


def _normalize_data_mask(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    mask = mask.to(device).float()
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    return mask


def _normalize_bit_mask(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    mask = mask.to(device).float()
    if mask.dim() == 1:
        mask = mask.view(1, -1, 1, 1)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    return mask
