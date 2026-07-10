from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from deeprx.matlab_bridge import (
    MatlabDeepRxBridge,
    OfficialBatch,
    PaperFigure6Config,
    paper_dataset_iteration,
    paper_dataset_split_frame_counts,
    sample_paper_dataset_parameters,
)


CACHE_VERSION = 1
CACHE_FILES = {
    "inputs": "inputs.npy",
    "target_bits": "target_bits.npy",
    "data_mask": "data_mask.npy",
    "bit_mask": "bit_mask.npy",
}
METADATA_FILE = "metadata.json"


class PaperTrainingCache:
    """Memory-mapped fixed train-frame cache for the paper dataset protocol."""

    def __init__(self, cache_dir: Path, metadata: dict, *, writable: bool = False):
        self.cache_dir = Path(cache_dir)
        self.metadata = metadata
        mmap_mode = "r+" if writable else "r"
        self._inputs = np.load(self.cache_dir / CACHE_FILES["inputs"], mmap_mode=mmap_mode)
        self._target_bits = np.load(self.cache_dir / CACHE_FILES["target_bits"], mmap_mode=mmap_mode)
        self._data_mask = np.load(self.cache_dir / CACHE_FILES["data_mask"], mmap_mode=mmap_mode)
        self._bit_mask = np.load(self.cache_dir / CACHE_FILES["bit_mask"], mmap_mode=mmap_mode)
        for key, array in (
            ("inputs", self._inputs),
            ("target_bits", self._target_bits),
            ("data_mask", self._data_mask),
            ("bit_mask", self._bit_mask),
        ):
            expected_shape = tuple(metadata["shapes"][key])
            if tuple(array.shape) != expected_shape or array.dtype != np.float32:
                raise ValueError(
                    f"Training cache {key} has shape/dtype {array.shape}/{array.dtype}, "
                    f"expected {expected_shape}/float32"
                )

    @classmethod
    def create(
        cls,
        cache_dir: str | Path,
        *,
        config: PaperFigure6Config,
        seed: int,
        frame_count: int,
        sample_batch: OfficialBatch,
        overwrite: bool = False,
    ) -> "PaperTrainingCache":
        cache_path = Path(cache_dir)
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        cache_path.mkdir(parents=True, exist_ok=True)
        _clear_known_cache_files(cache_path, overwrite=overwrite)

        metadata = {
            "version": CACHE_VERSION,
            "complete": False,
            "completed_frames": 0,
            "seed": int(seed),
            "frame_count": int(frame_count),
            "config": _config_payload(config),
            "files": CACHE_FILES,
            "shapes": {
                "inputs": [frame_count, *sample_batch.inputs.shape],
                "target_bits": [frame_count, *sample_batch.target_bits.shape],
                "data_mask": [frame_count, *sample_batch.data_mask.shape],
                "bit_mask": list(sample_batch.bit_mask.shape),
            },
            "dtypes": {
                "inputs": "float32",
                "target_bits": "float32",
                "data_mask": "float32",
                "bit_mask": "float32",
            },
        }
        _write_metadata(cache_path, metadata)
        for key in ("inputs", "target_bits", "data_mask", "bit_mask"):
            np.lib.format.open_memmap(
                cache_path / CACHE_FILES[key],
                mode="w+",
                dtype=np.float32,
                shape=tuple(metadata["shapes"][key]),
            )
        cache = cls(cache_path, metadata, writable=True)
        cache._bit_mask[...] = _to_float32_numpy(sample_batch.bit_mask)
        cache.flush()
        return cache

    @classmethod
    def open(
        cls,
        cache_dir: str | Path,
        *,
        config: PaperFigure6Config | None = None,
        seed: int | None = None,
        require_complete: bool = True,
        writable: bool = False,
    ) -> "PaperTrainingCache":
        cache_path = Path(cache_dir)
        metadata = read_cache_metadata(cache_path)
        validate_cache_metadata(metadata, config=config, seed=seed, require_complete=require_complete)
        return cls(cache_path, metadata, writable=writable)

    @property
    def frame_count(self) -> int:
        return int(self.metadata["frame_count"])

    @property
    def completed_frames(self) -> int:
        return int(self.metadata.get("completed_frames", 0))

    def write_frame(self, index: int, batch: OfficialBatch) -> None:
        if index < 0 or index >= self.frame_count:
            raise IndexError(f"frame index {index} is outside cache frame_count={self.frame_count}")
        _assert_shape("inputs", batch.inputs, self._inputs.shape[1:])
        _assert_shape("target_bits", batch.target_bits, self._target_bits.shape[1:])
        _assert_shape("data_mask", batch.data_mask, self._data_mask.shape[1:])
        self._inputs[index] = _to_float32_numpy(batch.inputs)
        self._target_bits[index] = _to_float32_numpy(batch.target_bits)
        self._data_mask[index] = _to_float32_numpy(batch.data_mask)

    def read_frames(self, indices: Iterable[int]) -> OfficialBatch:
        frame_indices = [int(index) % self.frame_count for index in indices]
        if not frame_indices:
            raise ValueError("At least one frame index is required")
        inputs = np.concatenate([self._inputs[index] for index in frame_indices], axis=0)
        target_bits = np.concatenate([self._target_bits[index] for index in frame_indices], axis=0)
        data_mask = np.concatenate([self._data_mask[index] for index in frame_indices], axis=0)
        bit_mask = np.array(self._bit_mask, dtype=np.float32, copy=True)
        return OfficialBatch(
            inputs=torch.from_numpy(inputs),
            target_bits=torch.from_numpy(target_bits),
            data_mask=torch.from_numpy(data_mask),
            bit_mask=torch.from_numpy(bit_mask),
        )

    def update_completed_frames(self, completed_frames: int) -> None:
        self.metadata["completed_frames"] = int(completed_frames)
        _write_metadata(self.cache_dir, self.metadata)

    def mark_complete(self) -> None:
        self.flush()
        self.metadata["completed_frames"] = self.frame_count
        self.metadata["complete"] = True
        _write_metadata(self.cache_dir, self.metadata)

    def flush(self) -> None:
        for array in (self._inputs, self._target_bits, self._data_mask, self._bit_mask):
            array.flush()


class CachedPaperTrainingStepDataset(Dataset):
    """Deterministic step dataset: step k reads frames k*n_frames ... k*n_frames+n_frames-1."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        start_step: int,
        end_step: int,
        n_frames: int,
        config: PaperFigure6Config,
        seed: int,
    ):
        if end_step < start_step:
            raise ValueError("end_step must be >= start_step")
        if n_frames <= 0:
            raise ValueError("n_frames must be positive")
        self.cache_dir = Path(cache_dir)
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.n_frames = int(n_frames)
        self.config = config
        self.seed = int(seed)
        self.metadata = read_cache_metadata(self.cache_dir)
        validate_cache_metadata(self.metadata, config=config, seed=seed, require_complete=True)
        self._cache: PaperTrainingCache | None = None

    def __len__(self) -> int:
        return self.end_step - self.start_step

    def __getitem__(self, local_index: int):
        step = self.start_step + int(local_index)
        start_frame = step * self.n_frames
        indices = [start_frame + offset for offset in range(self.n_frames)]
        batch = self._open_cache().read_frames(indices)
        return step, batch.inputs, batch.target_bits, batch.data_mask, batch.bit_mask

    def _open_cache(self) -> PaperTrainingCache:
        if self._cache is None:
            self._cache = PaperTrainingCache.open(
                self.cache_dir,
                config=self.config,
                seed=self.seed,
                require_complete=True,
            )
        return self._cache


def cached_paper_training_batch_iterator(
    cache_dir: str | Path,
    *,
    config: PaperFigure6Config,
    seed: int,
    start_step: int,
    end_step: int,
    n_frames: int,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    dataset = CachedPaperTrainingStepDataset(
        cache_dir,
        start_step=start_step,
        end_step=end_step,
        n_frames=n_frames,
        config=config,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    for step, inputs, target_bits, data_mask, bit_mask in loader:
        yield int(step), OfficialBatch(
            inputs=inputs,
            target_bits=target_bits,
            data_mask=data_mask,
            bit_mask=bit_mask,
        )


def build_paper_training_cache(
    bridge: MatlabDeepRxBridge,
    cache_dir: str | Path,
    *,
    config: PaperFigure6Config,
    seed: int,
    frame_count: int | None = None,
    overwrite: bool = False,
    progress_every: int = 100,
    progress_callback: Callable[[int, int], None] | None = None,
) -> PaperTrainingCache:
    train_frames, _ = paper_dataset_split_frame_counts(config)
    total_frames = train_frames if frame_count is None else int(frame_count)
    if total_frames <= 0:
        raise ValueError("frame_count must be positive")
    if total_frames > train_frames:
        raise ValueError(f"frame_count={total_frames} exceeds train split frame count {train_frames}")

    cache_path = Path(cache_dir)
    metadata_path = cache_path / METADATA_FILE
    if metadata_path.exists() and not overwrite:
        cache = PaperTrainingCache.open(
            cache_path,
            config=config,
            seed=seed,
            require_complete=False,
            writable=True,
        )
        if cache.frame_count != total_frames:
            raise ValueError(
                f"Existing cache frame_count={cache.frame_count} does not match requested frame_count={total_frames}"
            )
        if cache.metadata.get("complete", False):
            _report_progress(progress_callback, total_frames, total_frames)
            return cache
        start_frame = cache.completed_frames
        if start_frame < 0 or start_frame > total_frames:
            raise ValueError(f"Invalid completed_frames={start_frame} in training cache metadata")
    else:
        first_batch = _generate_cache_frame(bridge, config, seed=seed, frame_index=0)
        cache = PaperTrainingCache.create(
            cache_path,
            config=config,
            seed=seed,
            frame_count=total_frames,
            sample_batch=first_batch,
            overwrite=overwrite,
        )
        cache.write_frame(0, first_batch)
        cache.flush()
        cache.update_completed_frames(1)
        _report_progress(progress_callback, 1, total_frames)
        start_frame = 1

    for frame_index in range(start_frame, total_frames):
        cache.write_frame(frame_index, _generate_cache_frame(bridge, config, seed=seed, frame_index=frame_index))
        if progress_every > 0 and (frame_index + 1) % progress_every == 0:
            cache.flush()
            cache.update_completed_frames(frame_index + 1)
            _report_progress(progress_callback, frame_index + 1, total_frames)

    cache.mark_complete()
    _report_progress(progress_callback, total_frames, total_frames)
    return cache


def read_cache_metadata(cache_dir: str | Path) -> dict:
    path = Path(cache_dir) / METADATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"Training cache metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_cache_metadata(
    metadata: dict,
    *,
    config: PaperFigure6Config | None,
    seed: int | None,
    require_complete: bool,
) -> None:
    if metadata.get("version") != CACHE_VERSION:
        raise ValueError(f"Unsupported training cache version: {metadata.get('version')}")
    if require_complete and not metadata.get("complete", False):
        raise ValueError("Training cache is incomplete")
    if seed is not None and int(metadata.get("seed")) != int(seed):
        raise ValueError(f"Training cache seed {metadata.get('seed')} does not match requested seed {seed}")
    if config is not None and metadata.get("config") != _config_payload(config):
        raise ValueError("Training cache config does not match the requested paper configuration")


def _generate_cache_frame(
    bridge: MatlabDeepRxBridge,
    config: PaperFigure6Config,
    *,
    seed: int,
    frame_index: int,
) -> OfficialBatch:
    parameters = sample_paper_dataset_parameters(config, split="train", index=frame_index, seed=seed)
    iteration = paper_dataset_iteration(config, split="train", index=frame_index)
    return bridge.generate_training_batch(parameters, iteration=iteration, n_frames=1)


def _clear_known_cache_files(cache_dir: Path, *, overwrite: bool) -> None:
    existing = [cache_dir / METADATA_FILE, *(cache_dir / name for name in CACHE_FILES.values())]
    present = [path for path in existing if path.exists()]
    if present and not overwrite:
        raise FileExistsError(f"Training cache already exists at {cache_dir}; pass overwrite=True to replace it")
    for path in present:
        path.unlink()


def _config_payload(config: PaperFigure6Config) -> dict:
    return json.loads(json.dumps(asdict(config)))


def _write_metadata(cache_dir: Path, metadata: dict) -> None:
    path = cache_dir / METADATA_FILE
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary.replace(path)


def _to_float32_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _assert_shape(name: str, tensor: torch.Tensor, expected_shape) -> None:
    if tuple(tensor.shape) != tuple(expected_shape):
        raise ValueError(f"{name} shape {tuple(tensor.shape)} does not match cache shape {tuple(expected_shape)}")


def _report_progress(callback: Callable[[int, int], None] | None, current: int, total: int) -> None:
    if callback is not None:
        callback(current, total)
