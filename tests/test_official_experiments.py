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


def _load_replot_script():
    script = ROOT / "scripts" / "replot_figure6a.py"
    spec = importlib.util.spec_from_file_location("replot_figure6a", script)
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


def test_figure6a_metrics_convert_from_per_re_to_paper_whole_band_snr_without_changing_ber():
    metrics = initialize_figure6a_metrics(
        checkpoint_path=Path("model.pth"),
        snr_values=[0.0, 21.0],
        samples_per_point=500,
        n_frames=1,
    )
    for index, values in enumerate(metrics["curves"].values(), start=1):
        values.extend([index / 10.0, index / 100.0])
    original_curves = json.loads(json.dumps(metrics["curves"]))

    converted = experiments.convert_figure6a_metrics_to_paper_snr(metrics)

    assert converted["snr_conversion"]["offset_db"] == pytest.approx(2.15115366957388)
    assert converted["matlab_per_re_snr_db"] == [0.0, 21.0]
    assert converted["paper_whole_band_snr_db"] == pytest.approx(
        [-2.15115366957388, 18.84884633042612]
    )
    assert converted["curves"] == original_curves
    assert metrics["curves"] == original_curves
    assert "paper_whole_band_snr_db" not in metrics


def test_paper_snr_replot_writes_separate_artifacts_without_changing_source(tmp_path):
    metrics = initialize_figure6a_metrics(
        checkpoint_path=Path("model.pth"),
        snr_values=[0.0, 3.0],
        samples_per_point=500,
        n_frames=1,
    )
    for index, values in enumerate(metrics["curves"].values(), start=1):
        values.extend([index / 10.0, index / 100.0])
    source_path = tmp_path / "figure6a_metrics.json"
    corrected_metrics_path = tmp_path / "figure6a_metrics_paper_snr.json"
    corrected_figure_path = tmp_path / "figure6a_uncoded_ber_paper_snr.png"
    source_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    source_bytes = source_path.read_bytes()

    corrected = experiments.write_paper_snr_figure6a_artifacts(
        source_path,
        corrected_metrics_path,
        corrected_figure_path,
    )

    assert source_path.read_bytes() == source_bytes
    assert corrected_metrics_path.is_file()
    assert corrected_figure_path.stat().st_size > 1000
    assert corrected["paper_whole_band_snr_db"] == pytest.approx(
        [-2.15115366957388, 0.84884633042612]
    )
    assert json.loads(corrected_metrics_path.read_text(encoding="utf-8")) == corrected


def test_replot_cli_uses_separate_paper_snr_output_names(tmp_path):
    metrics = initialize_figure6a_metrics(
        checkpoint_path=Path("model.pth"),
        snr_values=[0.0, 3.0],
        samples_per_point=500,
        n_frames=1,
    )
    for index, values in enumerate(metrics["curves"].values(), start=1):
        values.extend([index / 10.0, index / 100.0])
    source_path = tmp_path / "figure6a_metrics.json"
    source_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    script = _load_replot_script()

    script.main(["--metrics", str(source_path)])

    assert (tmp_path / "figure6a_metrics_paper_snr.json").is_file()
    assert (tmp_path / "figure6a_uncoded_ber_paper_snr.png").is_file()


def test_figure6a_x_limits_leave_space_for_edge_markers():
    assert experiments.figure6a_x_limits([-2.15115366957388, 18.84884633042612]) == pytest.approx(
        (-2.65115366957388, 19.34884633042612)
    )


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
