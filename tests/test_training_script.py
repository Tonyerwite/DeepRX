import importlib.util
import inspect
from pathlib import Path

import torch

from deeprx.matlab_bridge import OfficialBatch, OfficialParameters


ROOT = Path(__file__).resolve().parents[1]


def _load_training_script():
    script = ROOT / "scripts" / "train_official_matlab.py"
    spec = importlib.util.spec_from_file_location("train_official_matlab", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_defaults_match_paper_scale():
    train = _load_training_script()
    args = train.build_arg_parser().parse_args([])

    assert args.steps == 30000
    assert args.n_frames == 8
    assert args.optimizer == "lamb"
    assert args.lr == 1e-2
    assert args.weight_decay == 1e-4
    assert args.warmup_steps == 800
    assert args.decay_start_fraction == 0.3


def test_training_uses_one_randomized_matlab_frame_per_paper_frame():
    train = _load_training_script()
    config = train.PaperFigure6Config()
    bridge = _RecordingBridge()
    rng = train.random.Random(2026)

    batch = train.generate_paper_training_batch(
        bridge,
        config,
        rng,
        step=0,
        n_frames=8,
        seed=2026,
    )

    assert len(bridge.calls) == 8
    assert {call["n_frames"] for call in bridge.calls} == {1}
    assert [call["iteration"] for call in bridge.calls] == list(range(1, 9))
    assert len({call["parameters"] for call in bridge.calls}) > 1
    assert batch.inputs.shape[0] == 80
    assert batch.target_bits.shape[0] == 80
    assert batch.data_mask.shape[0] == 80


def test_training_script_does_not_apply_unreported_gradient_clipping():
    train = _load_training_script()

    assert "clip_grad_norm_" not in inspect.getsource(train)


def test_paper_learning_rate_schedule_warms_up_then_decays():
    train = _load_training_script()

    assert train.paper_learning_rate(0, total_steps=30000, base_lr=1e-2, warmup_steps=800, decay_start_fraction=0.3) == 1.25e-5
    assert train.paper_learning_rate(799, total_steps=30000, base_lr=1e-2, warmup_steps=800, decay_start_fraction=0.3) == 1e-2
    assert train.paper_learning_rate(8999, total_steps=30000, base_lr=1e-2, warmup_steps=800, decay_start_fraction=0.3) == 1e-2
    assert train.paper_learning_rate(29999, total_steps=30000, base_lr=1e-2, warmup_steps=800, decay_start_fraction=0.3) < 1e-6


def test_lamb_optimizer_updates_parameter():
    train = _load_training_script()
    param = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = train.Lamb([param], lr=1e-2, weight_decay=1e-4)

    loss = (param**2).sum()
    loss.backward()
    before = param.detach().clone()
    optimizer.step()

    assert not torch.equal(before, param.detach())


class _RecordingBridge:
    def __init__(self):
        self.calls = []

    def generate_training_batch(self, parameters: OfficialParameters, *, iteration: int, n_frames: int):
        self.calls.append({"parameters": parameters, "iteration": iteration, "n_frames": n_frames})
        slot_count = n_frames * 10
        offset = float(iteration)
        return OfficialBatch(
            inputs=torch.full((slot_count, 10, 3, 2), offset),
            target_bits=torch.zeros((slot_count, 4, 3, 2)),
            data_mask=torch.ones((slot_count, 1, 3, 2)),
            bit_mask=torch.ones((1, 4, 1, 1)),
        )
