import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_workspace_does_not_keep_legacy_python_pipeline_modules():
    legacy_paths = [
        ROOT / "src" / "deeprx" / "data.py",
        ROOT / "src" / "deeprx" / "experiments.py",
        ROOT / "src" / "deeprx" / "ofdm.py",
        ROOT / "src" / "deeprx" / "qam.py",
        ROOT / "src" / "deeprx" / "receiver.py",
        ROOT / "tests" / "test_pipeline.py",
        ROOT / "tests" / "test_signal_chain.py",
    ]

    assert not [path for path in legacy_paths if path.exists()]


def test_config_uses_shallow_official_asset_root():
    config = json.loads((ROOT / "config" / "deeprx_paths.json").read_text(encoding="utf-8"))
    encoded = json.dumps(config, ensure_ascii=False)

    assert "vendor/mathworks/examples" not in encoded.replace("\\", "/")
    assert "mathworks_assets" not in config
    assert (ROOT / "official").is_dir()
