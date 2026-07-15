import importlib.util
from pathlib import Path

import numpy as np
import torch

from deeprx.matlab_bridge import OfficialBatch, PaperFigure6Config
from deeprx.training_cache import (
    CachedPaperTrainingStepDataset,
    PaperTrainingCache,
    build_paper_training_cache,
    cached_paper_training_batch_iterator,
)


ROOT = Path(__file__).resolve().parents[1]


def test_training_cache_builder_defaults_to_repo_data_directory():
    script = ROOT / "scripts" / "build_training_cache.py"
    spec = importlib.util.spec_from_file_location("build_training_cache", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_arg_parser().parse_args([])

    assert Path(args.cache_dir) == ROOT / "data" / "paper_train_cache"


def test_paper_training_cache_reads_steps_in_online_frame_order(tmp_path):
    config = PaperFigure6Config()
    sample = _frame_batch(0)
    cache = PaperTrainingCache.create(
        tmp_path,
        config=config,
        seed=2026,
        frame_count=4,
        sample_batch=sample,
        overwrite=True,
    )
    for frame_index in range(4):
        cache.write_frame(frame_index, _frame_batch(frame_index))
    cache.mark_complete()

    dataset = CachedPaperTrainingStepDataset(
        tmp_path,
        start_step=0,
        end_step=3,
        n_frames=2,
        config=config,
        seed=2026,
    )

    step0, inputs0, targets0, data_mask0, bit_mask0 = dataset[0]
    step1, inputs1, targets1, _, _ = dataset[1]
    step2, inputs2, targets2, _, _ = dataset[2]

    assert step0 == 0
    assert inputs0.shape == (20, 2, 3, 4)
    assert targets0.shape == (20, 4, 3, 4)
    assert data_mask0.shape == (20, 1, 3, 4)
    assert bit_mask0.shape == (1, 4, 1, 1)
    assert torch.equal(inputs0[:10], _frame_batch(0).inputs)
    assert torch.equal(inputs0[10:], _frame_batch(1).inputs)
    assert torch.equal(targets0[10:], _frame_batch(1).target_bits)

    assert step1 == 1
    assert torch.equal(inputs1[:10], _frame_batch(2).inputs)
    assert torch.equal(inputs1[10:], _frame_batch(3).inputs)
    assert torch.equal(targets1[10:], _frame_batch(3).target_bits)

    assert step2 == 2
    assert torch.equal(inputs2[:10], _frame_batch(0).inputs)
    assert torch.equal(inputs2[10:], _frame_batch(1).inputs)
    assert torch.equal(targets2[10:], _frame_batch(1).target_bits)


def test_paper_training_cache_reads_contiguous_frames_without_concatenating(tmp_path, monkeypatch):
    config = PaperFigure6Config()
    cache = PaperTrainingCache.create(
        tmp_path,
        config=config,
        seed=2026,
        frame_count=4,
        sample_batch=_frame_batch(0),
        overwrite=True,
    )
    for frame_index in range(4):
        cache.write_frame(frame_index, _frame_batch(frame_index))
    cache.mark_complete()

    def fail_concatenate(*args, **kwargs):
        raise AssertionError("contiguous cache reads must not allocate through np.concatenate")

    monkeypatch.setattr(np, "concatenate", fail_concatenate)
    batch = cache.read_frames([1, 2])

    assert batch.inputs.shape == (20, 2, 3, 4)
    assert torch.equal(batch.inputs[:10], _frame_batch(1).inputs)
    assert torch.equal(batch.inputs[10:], _frame_batch(2).inputs)


def test_cached_training_batch_iterator_yields_official_batches(tmp_path):
    config = PaperFigure6Config()
    cache = PaperTrainingCache.create(
        tmp_path,
        config=config,
        seed=2026,
        frame_count=2,
        sample_batch=_frame_batch(0),
        overwrite=True,
    )
    cache.write_frame(0, _frame_batch(0))
    cache.write_frame(1, _frame_batch(1))
    cache.mark_complete()

    batches = list(
        cached_paper_training_batch_iterator(
            tmp_path,
            config=config,
            seed=2026,
            start_step=1,
            end_step=2,
            n_frames=2,
            num_workers=0,
            pin_memory=False,
        )
    )

    assert len(batches) == 1
    step, batch = batches[0]
    assert step == 1
    assert isinstance(batch, OfficialBatch)
    assert torch.equal(batch.inputs[:10], _frame_batch(0).inputs)
    assert torch.equal(batch.inputs[10:], _frame_batch(1).inputs)


def test_cache_builder_resumes_from_last_flushed_frame(tmp_path):
    config = PaperFigure6Config()
    cache = PaperTrainingCache.create(
        tmp_path,
        config=config,
        seed=2026,
        frame_count=4,
        sample_batch=_frame_batch(0),
        overwrite=True,
    )
    cache.write_frame(0, _frame_batch(0))
    cache.write_frame(1, _frame_batch(1))
    cache.flush()
    cache.update_completed_frames(2)
    bridge = _RecordingCacheBridge()

    resumed = build_paper_training_cache(
        bridge,
        tmp_path,
        config=config,
        seed=2026,
        frame_count=4,
        progress_every=1,
    )

    assert bridge.iterations == [3, 4]
    assert resumed.metadata["complete"] is True
    assert resumed.completed_frames == 4
    batch = resumed.read_frames([0, 1, 2, 3])
    assert torch.equal(batch.inputs[:10], _frame_batch(0).inputs)
    assert torch.equal(batch.inputs[10:20], _frame_batch(1).inputs)
    assert torch.equal(batch.inputs[20:30], _frame_batch(2).inputs)
    assert torch.equal(batch.inputs[30:], _frame_batch(3).inputs)


def _frame_batch(frame_index: int) -> OfficialBatch:
    return OfficialBatch(
        inputs=torch.full((10, 2, 3, 4), float(frame_index), dtype=torch.float32),
        target_bits=torch.full((10, 4, 3, 4), float(frame_index % 2), dtype=torch.float32),
        data_mask=torch.ones((10, 1, 3, 4), dtype=torch.float32) * float(frame_index + 1),
        bit_mask=torch.ones((1, 4, 1, 1), dtype=torch.float32),
    )


class _RecordingCacheBridge:
    def __init__(self):
        self.iterations = []

    def generate_training_batch(self, parameters, *, iteration: int, n_frames: int):
        self.iterations.append(iteration)
        return _frame_batch(iteration - 1)
