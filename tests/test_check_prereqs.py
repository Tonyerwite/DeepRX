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
    example_root = tmp_path / "examples"
    matlab_example_dir = example_root / "AINativeFullyConvolutionalReceiverExample"
    pytorch_example_dir = example_root / "VerifyAINativeReceiverUsingMATLABPytorchCoexecutionExample"
    mat_file = tmp_path / "DeepRx_2M.mat"
    pth_file = tmp_path / "deeprx_30k.pth"
    torch_file = tmp_path / "deeprx_model.py"
    helper_file = tmp_path / "hCreateDeepRx.m"

    for file_path in [matlab, python, mat_file, pth_file, torch_file, helper_file]:
        file_path.write_text("x", encoding="utf-8")
    for dir_path in [support_dir, example_root, matlab_example_dir, pytorch_example_dir]:
        dir_path.mkdir()

    return {
        "matlab": {"r2025b_executable": str(matlab)},
        "python": {"venv_python": str(python)},
        "mathworks_assets": {
            "six_g_support_package_path": str(support_dir),
            "official_example_root": str(example_root),
            "matlab_deeprx_example_dir": str(matlab_example_dir),
            "pytorch_coexecution_example_dir": str(pytorch_example_dir),
            "deep_rx_2m_mat": str(mat_file),
            "deeprx_30k_pth": str(pth_file),
            "official_matlab_helper_files": [str(helper_file)],
            "official_pytorch_files": [str(torch_file)],
        },
    }


def test_iter_configured_path_checks_includes_official_deeprx_assets(tmp_path):
    data = _sample_config(tmp_path)
    check_prereqs = _load_check_prereqs()
    checks = list(check_prereqs.iter_configured_path_checks(data))
    names = {check.name for check in checks}

    assert "MATLAB R2025b executable" in names
    assert "6G support package path" in names
    assert "Official DeepRx_2M.mat" in names
    assert "Official deeprx_30k.pth" in names
    assert "Official MATLAB helper file: hCreateDeepRx.m" in names
    assert "Official PyTorch reference file: deeprx_model.py" in names
    assert all(check.exists for check in checks)
