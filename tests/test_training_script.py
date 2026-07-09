import importlib.util
from pathlib import Path

import torch


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
