import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATHS = ROOT / "config" / "deeprx_paths.json"


def main():
    data = json.loads(PATHS.read_text(encoding="utf-8"))
    matlab = Path(data["matlab"]["r2025b_executable"])
    python = Path(data["python"]["venv_python"])
    signpost = Path(data["mathworks_assets"]["six_g_exploration_library_signpost"])

    print(f"MATLAB R2025b executable: {matlab} exists={matlab.exists()}")
    print(f"Python venv executable: {python} exists={python.exists()}")
    print(f"6G signpost: {signpost} exists={signpost.exists()} size={signpost.stat().st_size if signpost.exists() else 0}")

    if python.exists():
        cmd = [str(python), "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"]
        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        print("PyTorch check:")
        print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())


if __name__ == "__main__":
    main()
