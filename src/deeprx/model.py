from typing import Tuple

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


class MathWorksResidualBlock(nn.Module):
    def __init__(
        self,
        idx: int,
        channels: Tuple[int, ...],
        dilations: Tuple[Tuple[int, int], ...],
        is_projection: bool = False,
    ):
        super().__init__()
        in_channel = channels[idx - 1]
        prev_channel = channels[idx - 2] if is_projection else in_channel
        dilation_factor = dilations[idx - 1]

        self.relu = nn.ReLU()
        self.relu_res = nn.ReLU()
        self.bn = nn.BatchNorm2d(in_channel)
        self.bn_res = nn.BatchNorm2d(prev_channel)
        self.conv1_3x3sep = nn.Conv2d(
            prev_channel,
            2 * prev_channel,
            kernel_size=3,
            groups=prev_channel,
            dilation=dilation_factor,
            stride=1,
            padding="same",
        )
        self.conv2_1x1 = nn.Conv2d(2 * prev_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.conv3_3x3sep = nn.Conv2d(
            in_channel,
            2 * in_channel,
            kernel_size=3,
            groups=in_channel,
            dilation=dilation_factor,
            stride=1,
            padding="same",
        )
        self.conv4_1x1 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.projection = is_projection
        if self.projection:
            self.shortcut = nn.Conv2d(prev_channel, in_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.projection:
            y = self.bn_res(x)
            z = self.relu_res(y)
            y = self.conv1_3x3sep(z)
            y = self.conv2_1x1(y)
            y = self.bn(y)
            y = self.relu(y)
            y = self.conv3_3x3sep(y)
            y = self.conv4_1x1(y)
            shortcut = self.shortcut(z)
        else:
            y = self.bn_res(x)
            y = self.relu_res(y)
            y = self.conv1_3x3sep(y)
            y = self.conv2_1x1(y)
            y = self.bn(y)
            y = self.relu(y)
            y = self.conv3_3x3sep(y)
            y = self.conv4_1x1(y)
            shortcut = x
        return y + shortcut


class DeepRx(nn.Module):
    """MathWorks-compatible PyTorch DeepRx network for the Honkala et al. receiver."""

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
        if depth_multiplier != 2:
            raise ValueError("The MathWorks-compatible DeepRx uses depth_multiplier=2.")
        self.n_rx_antennas = n_rx_antennas
        self.n_rx = n_rx_antennas
        self.max_bits_per_symbol = max_bits_per_symbol
        n_input_channels = 2 * (2 * n_rx_antennas + 1)
        channels = tuple(out_channels for out_channels, _ in self.block_configs)
        dilations = tuple(dilation for _, dilation in self.block_configs)

        self.conv_in = nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding="same")
        self.resnet_block_1 = MathWorksResidualBlock(1, channels, dilations, is_projection=False)
        self.resnet_block_2 = MathWorksResidualBlock(2, channels, dilations, is_projection=False)
        self.resnet_block_3 = MathWorksResidualBlock(3, channels, dilations, is_projection=True)
        self.resnet_block_4 = MathWorksResidualBlock(4, channels, dilations, is_projection=False)
        self.resnet_block_5 = MathWorksResidualBlock(5, channels, dilations, is_projection=True)
        self.resnet_block_6 = MathWorksResidualBlock(6, channels, dilations, is_projection=False)
        self.resnet_block_7 = MathWorksResidualBlock(7, channels, dilations, is_projection=False)
        self.resnet_block_8 = MathWorksResidualBlock(8, channels, dilations, is_projection=True)
        self.resnet_block_9 = MathWorksResidualBlock(9, channels, dilations, is_projection=False)
        self.resnet_block_10 = MathWorksResidualBlock(10, channels, dilations, is_projection=True)
        self.resnet_block_11 = MathWorksResidualBlock(11, channels, dilations, is_projection=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv_out = nn.Conv2d(64, max_bits_per_symbol, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(z)
        out = self.resnet_block_1(out)
        out = self.resnet_block_2(out)
        out = self.resnet_block_3(out)
        out = self.resnet_block_4(out)
        out = self.resnet_block_5(out)
        out = self.resnet_block_6(out)
        out = self.resnet_block_7(out)
        out = self.resnet_block_8(out)
        out = self.resnet_block_9(out)
        out = self.resnet_block_10(out)
        out = self.resnet_block_11(out)
        out = self.bn(out)
        out = self.relu(out)
        return self.conv_out(out)

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
