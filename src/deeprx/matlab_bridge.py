from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from deeprx.model import create_bit_mask


@dataclass(frozen=True)
class PaperFigure6Config:
    """Paper Table II and Fig. 6(a) settings for uncoded BER reproduction.

    The paper's original dataset seed and frame order are not public. This
    config models the described fixed 500k-TTI, 60/40 split with deterministic
    frame indices, while MATLAB still generates each indexed frame online.
    """

    dataset_ttis: int = 500_000
    ttis_per_frame: int = 10
    train_fraction: float = 0.6
    validation_samples_per_point: int = 500
    carrier_frequency_hz: float = 4.0e9
    n_size_grid: int = 26
    subcarrier_spacing_khz: int = 15
    modulation: str = "16QAM"
    code_rate: float = 658 / 1024
    n_frames: int = 1
    snr_limits_db: Tuple[float, float] = (-4.0, 32.0)
    delay_spread_limits_s: Tuple[float, float] = (10e-9, 300e-9)
    maximum_doppler_shift_limits_hz: Tuple[float, float] = (0.0, 500.0)
    train_channels: Tuple[str, ...] = ("CDL-B", "CDL-C", "CDL-D", "TDL-B", "TDL-C", "TDL-D")
    validation_channels: Tuple[str, ...] = ("CDL-A", "CDL-E", "TDL-A", "TDL-E")
    figure6_sinr_points_db: Tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0)


@dataclass(frozen=True)
class OfficialBatch:
    inputs: torch.Tensor
    target_bits: torch.Tensor
    data_mask: torch.Tensor
    bit_mask: torch.Tensor


@dataclass(frozen=True)
class OfficialParameters:
    snr_db: float
    channel_model: str
    delay_spread_s: float
    max_doppler_shift_hz: float
    dmrs_additional_position: int
    dmrs_configuration_type: int


@dataclass(frozen=True)
class MatlabBridgePaths:
    matlab_example_dir: Path
    pytorch_example_dir: Path
    python_executable: Path
    project_matlab_dir: Path
    default_checkpoint: Path


def pilot_count_to_dmrs_additional_position(pilot_count: int) -> int:
    if pilot_count == 1:
        return 0
    if pilot_count == 2:
        return 1
    raise ValueError(f"pilot_count must be 1 or 2, got {pilot_count}")


def paper_dataset_frame_count(config: PaperFigure6Config) -> int:
    if config.dataset_ttis % config.ttis_per_frame != 0:
        raise ValueError("dataset_ttis must be divisible by ttis_per_frame")
    return config.dataset_ttis // config.ttis_per_frame


def paper_dataset_split_frame_counts(config: PaperFigure6Config) -> Tuple[int, int]:
    total_frames = paper_dataset_frame_count(config)
    train_frames = int(total_frames * config.train_fraction)
    return train_frames, total_frames - train_frames


def paper_dataset_global_frame_index(config: PaperFigure6Config, *, split: str, index: int) -> int:
    train_frames, validation_frames = paper_dataset_split_frame_counts(config)
    if split == "train":
        return index % train_frames
    if split == "validation":
        return train_frames + (index % validation_frames)
    raise ValueError(f"split must be 'train' or 'validation', got {split!r}")


def paper_dataset_iteration(config: PaperFigure6Config, *, split: str, index: int) -> int:
    return paper_dataset_global_frame_index(config, split=split, index=index) + 1


def sample_paper_dataset_parameters(
    config: PaperFigure6Config,
    *,
    split: str,
    index: int,
    seed: int,
    snr_db: float | None = None,
    pilot_count: int | None = None,
) -> OfficialParameters:
    frame_index = paper_dataset_global_frame_index(config, split=split, index=index)
    rng = random.Random(seed * 1_000_003 + frame_index)

    if split == "train":
        channels = config.train_channels
        snr = rng.uniform(*config.snr_limits_db) if snr_db is None else snr_db
        dmrs_additional_position = rng.randint(0, 1)
    elif split == "validation":
        channels = config.validation_channels
        if snr_db is None:
            raise ValueError("snr_db is required for validation dataset sampling")
        if pilot_count is None:
            raise ValueError("pilot_count is required for validation dataset sampling")
        snr = snr_db
        dmrs_additional_position = pilot_count_to_dmrs_additional_position(pilot_count)
    else:
        raise ValueError(f"split must be 'train' or 'validation', got {split!r}")

    return OfficialParameters(
        snr_db=float(snr),
        channel_model=rng.choice(channels),
        delay_spread_s=rng.uniform(*config.delay_spread_limits_s),
        max_doppler_shift_hz=rng.uniform(*config.maximum_doppler_shift_limits_hz),
        dmrs_additional_position=dmrs_additional_position,
        dmrs_configuration_type=rng.randint(1, 2),
    )


def convert_official_batch_arrays(
    inputs_fscn: np.ndarray,
    targets_fsbn: np.ndarray,
    data_mask_fs1n: np.ndarray,
    modulation: str,
    max_bits_per_symbol: int,
) -> OfficialBatch:
    inputs = torch.from_numpy(np.asarray(inputs_fscn, dtype=np.float32)).permute(3, 2, 0, 1).contiguous()
    targets = torch.from_numpy(np.asarray(targets_fsbn, dtype=np.float32)).permute(3, 2, 0, 1).contiguous()
    data_mask = torch.from_numpy(np.asarray(data_mask_fs1n, dtype=np.float32)).permute(3, 2, 0, 1).contiguous()
    bit_mask = create_bit_mask(modulation, max_bits=max_bits_per_symbol, device="cpu")
    return OfficialBatch(inputs=inputs, target_bits=targets, data_mask=data_mask, bit_mask=bit_mask)


def load_matlab_bridge_paths(project_root: Path | None = None) -> MatlabBridgePaths:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    data = json.loads((root / "config" / "deeprx_paths.json").read_text(encoding="utf-8"))
    official_dir = root / "official"
    matlab_dir = root / "matlab"
    return MatlabBridgePaths(
        matlab_example_dir=matlab_dir,
        pytorch_example_dir=official_dir,
        python_executable=Path(data["python"]["venv_python"]),
        project_matlab_dir=matlab_dir,
        default_checkpoint=official_dir / "deeprx_30k.pth",
    )


def sample_official_parameters(
    config: PaperFigure6Config,
    rng: random.Random,
    *,
    mode: str,
    snr_db: float | None = None,
    pilot_count: int | None = None,
) -> OfficialParameters:
    if mode == "train":
        channels = config.train_channels
        snr = rng.uniform(*config.snr_limits_db) if snr_db is None else snr_db
        dmrs_additional_position = rng.randint(0, 1)
    elif mode == "validate":
        channels = config.validation_channels
        if snr_db is None:
            raise ValueError("snr_db is required for validation/evaluation sampling")
        snr = snr_db
        if pilot_count is None:
            raise ValueError("pilot_count is required for validation/evaluation sampling")
        dmrs_additional_position = pilot_count_to_dmrs_additional_position(pilot_count)
    else:
        raise ValueError(f"mode must be 'train' or 'validate', got {mode!r}")

    return OfficialParameters(
        snr_db=float(snr),
        channel_model=rng.choice(channels),
        delay_spread_s=rng.uniform(*config.delay_spread_limits_s),
        max_doppler_shift_hz=rng.uniform(*config.maximum_doppler_shift_limits_hz),
        dmrs_additional_position=dmrs_additional_position,
        dmrs_configuration_type=rng.randint(1, 2),
    )


class MatlabDeepRxBridge:
    """Thin Python wrapper around the official MathWorks DeepRx data path."""

    def __init__(self, paths: MatlabBridgePaths | None = None):
        self.paths = paths or load_matlab_bridge_paths()
        self._engine = None

    def __enter__(self) -> "MatlabDeepRxBridge":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self._engine is not None:
            return
        import matlab.engine

        self._engine = matlab.engine.start_matlab()
        self._engine.addpath(str(self.paths.matlab_example_dir), nargout=0)
        self._engine.addpath(str(self.paths.pytorch_example_dir), nargout=0)
        self._engine.addpath(str(self.paths.project_matlab_dir), nargout=0)
        self._engine.eval("try, parallel.gpu.enableCUDAForwardCompatibility(true); catch, end", nargout=0)

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def generate_training_batch(
        self,
        parameters: OfficialParameters,
        *,
        iteration: int,
        n_frames: int = 1,
        max_bits_per_symbol: int = 4,
    ) -> OfficialBatch:
        self.start()
        batch = self._engine.deeprx_export_official_batch(
            str(self.paths.matlab_example_dir),
            float(iteration),
            float(parameters.snr_db),
            parameters.channel_model,
            float(parameters.delay_spread_s),
            float(parameters.max_doppler_shift_hz),
            float(parameters.dmrs_additional_position),
            float(parameters.dmrs_configuration_type),
            float(n_frames),
            nargout=1,
        )
        return convert_official_batch_arrays(
            _matlab_array_to_numpy(_matlab_struct_field(batch, "X")),
            _matlab_array_to_numpy(_matlab_struct_field(batch, "TargetBits")),
            _matlab_array_to_numpy(_matlab_struct_field(batch, "DataMask")),
            "16QAM",
            max_bits_per_symbol=max_bits_per_symbol,
        )

    def evaluate_practical_lmmse(
        self,
        parameters: OfficialParameters,
        *,
        iteration: int,
        n_frames: int = 1,
    ) -> float:
        self.start()
        results = self._engine.deeprx_evaluate_practical_lmmse(
            str(self.paths.matlab_example_dir),
            float(iteration),
            float(parameters.snr_db),
            parameters.channel_model,
            float(parameters.delay_spread_s),
            float(parameters.max_doppler_shift_hz),
            float(parameters.dmrs_additional_position),
            float(parameters.dmrs_configuration_type),
            float(n_frames),
            nargout=1,
        )
        return float(_matlab_struct_field(results, "UncodedBER"))

    def evaluate_known_channel_lmmse(
        self,
        parameters: OfficialParameters,
        *,
        iteration: int,
        n_frames: int = 1,
    ) -> float:
        self.start()
        results = self._engine.deeprx_evaluate_known_channel_lmmse(
            str(self.paths.matlab_example_dir),
            float(iteration),
            float(parameters.snr_db),
            parameters.channel_model,
            float(parameters.delay_spread_s),
            float(parameters.max_doppler_shift_hz),
            float(parameters.dmrs_additional_position),
            float(parameters.dmrs_configuration_type),
            float(n_frames),
            nargout=1,
        )
        return float(_matlab_struct_field(results, "UncodedBER"))

    def evaluate_pytorch_deeprx(
        self,
        parameters: OfficialParameters,
        *,
        model_path: Path,
        iteration: int,
        n_frames: int = 1,
    ) -> float:
        self.start()
        python_path = self.paths.python_executable
        if python_path.name.lower() == "python.exe":
            candidate = python_path.with_name("pythonw.exe")
            if candidate.exists():
                python_path = candidate
        results = self._engine.deeprx_evaluate_pytorch_deeprx(
            str(self.paths.pytorch_example_dir),
            str(python_path),
            str(model_path),
            float(iteration),
            float(parameters.snr_db),
            parameters.channel_model,
            float(parameters.delay_spread_s),
            float(parameters.max_doppler_shift_hz),
            float(parameters.dmrs_additional_position),
            float(parameters.dmrs_configuration_type),
            float(n_frames),
            nargout=1,
        )
        return float(_matlab_struct_field(results, "UncodedBER"))


def _matlab_struct_field(value, field_name: str):
    try:
        return value[field_name]
    except (KeyError, TypeError):
        return getattr(value, field_name)


def _matlab_array_to_numpy(value) -> np.ndarray:
    return np.array(value, dtype=np.float32)
