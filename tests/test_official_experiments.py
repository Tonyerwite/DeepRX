import importlib.util
from pathlib import Path

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


def test_known_channel_lmmse_wrapper_uses_official_perfect_estimator_branch():
    wrapper = (ROOT / "matlab" / "deeprx_evaluate_known_channel_lmmse.m").read_text(encoding="utf-8")

    assert "PerfectChannelEstimator = true" in wrapper
    assert "hGetFeaturesAndLabels" in wrapper
