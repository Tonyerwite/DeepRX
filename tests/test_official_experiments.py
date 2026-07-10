import importlib.util
import json
from pathlib import Path

import pytest

import deeprx.official_experiments as experiments
from deeprx.official_experiments import initialize_figure6a_metrics


ROOT = Path(__file__).resolve().parents[1]


def _load_reproduction_script():
    script = ROOT / "scripts" / "reproduce_figure6a.py"
    spec = importlib.util.spec_from_file_location("reproduce_figure6a", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_figure6a_metrics_include_known_channel_lmmse_curve():
    metrics = initialize_figure6a_metrics(
        checkpoint_path=Path("model.pth"),
        snr_values=[0.0, 3.0],
        samples_per_point=500,
        n_frames=1,
    )

    assert set(metrics["curves"]) == {
        "deeprx_1_pilot",
        "deeprx_2_pilots",
        "lmmse_1_pilot",
        "lmmse_2_pilots",
        "lmmse_known_channel",
    }


def test_reproduce_figure6a_defaults_to_paper_scale_monte_carlo():
    script = _load_reproduction_script()
    args = script.build_arg_parser().parse_args([])

    assert args.samples_per_point == 500
    assert args.restart is False


def test_known_channel_lmmse_wrapper_uses_official_perfect_estimator_branch():
    wrapper = (ROOT / "matlab" / "deeprx_evaluate_known_channel_lmmse.m").read_text(encoding="utf-8")

    assert "PerfectChannelEstimator = true" in wrapper
    assert "hGetFeaturesAndLabels" in wrapper


def test_figure6a_resumes_after_completed_snr_point(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"model")
    first_bridge = _FakeEvaluationBridge(fail_snr=3.0)
    monkeypatch.setattr(experiments, "load_matlab_bridge_paths", lambda: None)
    monkeypatch.setattr(experiments, "MatlabDeepRxBridge", lambda paths: first_bridge)

    with pytest.raises(RuntimeError, match="simulated evaluation interruption"):
        experiments.run_figure6a_reproduction(
            checkpoint,
            tmp_path,
            snr_points=[0.0, 3.0],
            samples_per_point=1,
            n_frames=1,
            seed=2026,
        )

    progress_path = tmp_path / "figure6a_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["completed_snr_count"] == 1

    second_bridge = _FakeEvaluationBridge()
    monkeypatch.setattr(experiments, "MatlabDeepRxBridge", lambda paths: second_bridge)
    metrics = experiments.run_figure6a_reproduction(
        checkpoint,
        tmp_path,
        snr_points=[0.0, 3.0],
        samples_per_point=1,
        n_frames=1,
        seed=2026,
    )

    assert {call[1] for call in second_bridge.calls} == {3.0}
    assert all(len(values) == 2 for values in metrics["curves"].values())
    assert not progress_path.exists()


class _FakeEvaluationBridge:
    def __init__(self, fail_snr=None):
        self.fail_snr = fail_snr
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def _value(self, receiver, parameters):
        self.calls.append((receiver, parameters.snr_db, parameters.dmrs_additional_position))
        if self.fail_snr == parameters.snr_db:
            raise RuntimeError("simulated evaluation interruption")
        return 0.1 + parameters.snr_db / 100.0

    def evaluate_pytorch_deeprx(self, parameters, **kwargs):
        return self._value("deeprx", parameters)

    def evaluate_practical_lmmse(self, parameters, **kwargs):
        return self._value("practical", parameters)

    def evaluate_known_channel_lmmse(self, parameters, **kwargs):
        return self._value("known", parameters)
