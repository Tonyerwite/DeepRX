from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PaperConfig:
    """Default parameters collected from the DeepRx paper and MathWorks examples."""

    carrier_frequency_hz: float = 4.0e9
    mathworks_carrier_frequency_hz: float = 3.5e9
    n_prb: int = 26
    n_subcarriers: int = 312
    n_fft: int = 512
    subcarrier_spacing_khz: int = 15
    cp_length: int = 36
    n_symbols: int = 14
    tti_ms: float = 1.0
    modulation: str = "16QAM"
    code_rate: float = 658 / 1024
    n_tx_antennas: int = 1
    n_rx_antennas: int = 2
    snr_range_db: Tuple[float, float] = (-4.0, 32.0)
    delay_spread_range_s: Tuple[float, float] = (10e-9, 300e-9)
    doppler_range_hz: Tuple[float, float] = (0.0, 500.0)
    sir_range_db: Tuple[float, float] = (0.0, 36.0)
    train_channels: Tuple[str, ...] = ("CDL-B", "CDL-C", "CDL-D", "TDL-B", "TDL-C", "TDL-D")
    validation_channels: Tuple[str, ...] = ("CDL-A", "CDL-E", "TDL-A", "TDL-E")
    pilot_configs: Tuple[str, ...] = (
        "1_pilot_left",
        "1_pilot_right",
        "2_pilots_left",
        "2_pilots_right",
    )


PAPER_CONFIG = PaperConfig()
