import importlib.util
from pathlib import Path


def _load_check_prereqs():
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_prereqs.py"
    spec = importlib.util.spec_from_file_location("check_prereqs", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_config(tmp_path):
    matlab = tmp_path / "matlab.exe"
    python = tmp_path / "python.exe"
    support_dir = tmp_path / "pre6g"

    for file_path in [matlab, python]:
        file_path.write_text("x", encoding="utf-8")
    for dir_path in [support_dir]:
        dir_path.mkdir()

    return {
        "matlab": {"r2025b_executable": str(matlab)},
        "python": {"venv_python": str(python)},
        "matlab_support": {"six_g_support_package_path": str(support_dir)},
    }


def test_iter_configured_path_checks_includes_official_deeprx_assets(tmp_path):
    data = _sample_config(tmp_path)
    check_prereqs = _load_check_prereqs()
    checks = list(check_prereqs.iter_configured_path_checks(data))
    names = {check.name for check in checks}

    assert "MATLAB R2025b executable" in names
    assert "6G support package path" in names
    assert "Official deeprx_30k.pth" in names
    assert "MATLAB runtime file: hGetFeaturesAndLabels.m" in names
    assert "MATLAB runtime file: deeprx_evaluate_known_channel_lmmse.m" in names
    assert "Official PyTorch runtime file: deeprx_model.py" in names
