import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
PATHS = ROOT / "config" / "deeprx_paths.json"
MATLAB_DIR = ROOT / "matlab"
OFFICIAL_DIR = ROOT / "official"


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
    yield _path_check("MATLAB R2025b executable", data["matlab"]["r2025b_executable"])
    yield _path_check("Python venv executable", data["python"]["venv_python"])
    yield _path_check("6G support package path", data["matlab_support"]["six_g_support_package_path"])
    yield PathCheck("MATLAB helper dir", MATLAB_DIR, MATLAB_DIR.exists())
    yield PathCheck("Official PyTorch coexecution dir", OFFICIAL_DIR, OFFICIAL_DIR.exists())
    yield PathCheck("Official deeprx_30k.pth", OFFICIAL_DIR / "deeprx_30k.pth", (OFFICIAL_DIR / "deeprx_30k.pth").exists(), (OFFICIAL_DIR / "deeprx_30k.pth").stat().st_size if (OFFICIAL_DIR / "deeprx_30k.pth").is_file() else None)

    matlab_files = [
        "HARQEntity.m",
        "hGetAdditionalSystemParameters.m",
        "hGetFeaturesAndLabels.m",
        "deeprx_build_paper_sim_parameters.m",
        "deeprx_export_official_batch.m",
        "deeprx_evaluate_practical_lmmse.m",
        "deeprx_evaluate_pytorch_deeprx.m",
        "requirements_6GVerify_project.txt",
    ]
    for name in matlab_files:
        path = MATLAB_DIR / name
        yield PathCheck(f"MATLAB runtime file: {name}", path, path.exists(), path.stat().st_size if path.is_file() else None)

    official_files = [
        "deeprx.py",
        "deeprx_model.py",
        "hCreateTorchDeepRx.m",
        "helperLibraryChecker.m",
        "helperSetupPyenv.m",
        "helperinstalledlibs.py",
    ]
    for name in official_files:
        path = OFFICIAL_DIR / name
        yield PathCheck(f"Official PyTorch runtime file: {name}", path, path.exists(), path.stat().st_size if path.is_file() else None)


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
