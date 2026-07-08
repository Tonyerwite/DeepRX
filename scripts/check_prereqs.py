import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
PATHS = ROOT / "config" / "deeprx_paths.json"


@dataclass(frozen=True)
class PathCheck:
    name: str
    path: Path
    exists: bool
    size: Optional[int] = None


def _path_check(name: str, raw_path: str) -> PathCheck:
    path = Path(raw_path)
    exists = path.exists()
    size = path.stat().st_size if exists and path.is_file() else None
    return PathCheck(name=name, path=path, exists=exists, size=size)


def iter_configured_path_checks(data: Dict) -> Iterable[PathCheck]:
    assets = data["mathworks_assets"]

    yield _path_check("MATLAB R2025b executable", data["matlab"]["r2025b_executable"])
    yield _path_check("Python venv executable", data["python"]["venv_python"])
    yield _path_check("6G support package path", assets["six_g_support_package_path"])
    yield _path_check("Official example root", assets["official_example_root"])
    yield _path_check("Official MATLAB DeepRx example dir", assets["matlab_deeprx_example_dir"])
    yield _path_check("Official PyTorch coexecution example dir", assets["pytorch_coexecution_example_dir"])
    yield _path_check("Official DeepRx_2M.mat", assets["deep_rx_2m_mat"])
    yield _path_check("Official deeprx_30k.pth", assets["deeprx_30k_pth"])

    for raw_path in assets.get("official_matlab_helper_files", []):
        yield _path_check(f"Official MATLAB helper file: {Path(raw_path).name}", raw_path)

    for raw_path in assets.get("official_pytorch_files", []):
        yield _path_check(f"Official PyTorch reference file: {Path(raw_path).name}", raw_path)


def _print_path_checks(data: Dict) -> int:
    missing = 0
    print("Configured path checks:")
    for check in iter_configured_path_checks(data):
        status = "OK" if check.exists else "MISSING"
        suffix = f" size={check.size}" if check.size is not None else ""
        print(f"- {status}: {check.name}: {check.path}{suffix}")
        missing += 0 if check.exists else 1
    return missing


def _run_pytorch_check(python: Path) -> int:
    if not python.exists():
        return 1

    cmd = [
        str(python),
        "-c",
        (
            "import torch; "
            "print('TORCH_VERSION=' + torch.__version__); "
            "print('TORCH_CUDA_AVAILABLE=' + str(torch.cuda.is_available())); "
            "print('TORCH_GPU=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda'))"
        ),
    ]
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    print("\nPyTorch check:")
    print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    return result.returncode


def _run_matlab_check(matlab: Path) -> int:
    if not matlab.exists():
        return 1

    matlab_code = (
        "try; "
        "fprintf('MATLAB_VERSION=%s\\n',version); "
        "addons=matlab.addons.installedAddons; "
        "hit=contains(addons.Identifier,'NRX5G') | contains(addons.Name,'6G','IgnoreCase',true); "
        "fprintf('NRX5G_INSTALLED=%d\\n',any(hit)); "
        "if any(hit), disp(addons(hit,:)); end; "
        "parallel.gpu.enableCUDAForwardCompatibility(true); "
        "fprintf('CAN_USE_GPU=%d\\n',canUseGPU); "
        "if canUseGPU, g=gpuDevice; fprintf('GPU_NAME=%s\\n',g.Name); "
        "fprintf('GPU_COMPUTE_CAPABILITY=%s\\n',g.ComputeCapability); end; "
        "fprintf('CAN_USE_PARPOOL=%d\\n',canUseParallelPool); "
        "catch ME; fprintf(2,'MATLAB_CHECK_ERROR:%s:%s\\n',ME.identifier,ME.message); exit(1); end"
    )
    result = subprocess.run(
        [str(matlab), "-batch", matlab_code],
        text=True,
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        timeout=240,
    )
    print("\nMATLAB check:")
    print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    return result.returncode


def main():
    data = json.loads(PATHS.read_text(encoding="utf-8"))
    matlab = Path(data["matlab"]["r2025b_executable"])
    python = Path(data["python"]["venv_python"])

    missing = _print_path_checks(data)
    pytorch_rc = _run_pytorch_check(python)
    matlab_rc = _run_matlab_check(matlab)

    if missing or pytorch_rc or matlab_rc:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
