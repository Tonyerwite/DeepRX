from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from deeprx.matlab_bridge import (
    MatlabDeepRxBridge,
    PaperFigure6Config,
    load_matlab_bridge_paths,
    paper_dataset_iteration,
    sample_paper_dataset_parameters,
)


def run_figure6a_reproduction(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    snr_points: Iterable[float] | None = None,
    samples_per_point: int | None = None,
    n_frames: int = 1,
    seed: int = 2026,
    restart: bool = False,
) -> Dict:
    config = PaperFigure6Config()
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir)
    snr_values = list(config.figure6_sinr_points_db if snr_points is None else snr_points)
    samples = config.validation_samples_per_point if samples_per_point is None else samples_per_point
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "figure6a_progress.json"
    signature = {
        "checkpoint": str(checkpoint_path),
        "snr_db": snr_values,
        "samples_per_point": samples,
        "n_frames": n_frames,
        "seed": seed,
    }
    if restart:
        progress_path.unlink(missing_ok=True)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("signature") != signature:
            raise ValueError("Existing Fig. 6(a) progress does not match this run; pass restart=True to replace it")
        metrics = progress["metrics"]
        completed_snr_count = int(progress["completed_snr_count"])
    else:
        metrics = initialize_figure6a_metrics(
            checkpoint_path=checkpoint_path,
            snr_values=snr_values,
            samples_per_point=samples,
            n_frames=n_frames,
        )
        completed_snr_count = 0

    with MatlabDeepRxBridge(load_matlab_bridge_paths()) as bridge:
        for snr_index, snr in enumerate(snr_values[completed_snr_count:], start=completed_snr_count):
            known_channel_values: List[float] = []
            for pilot_count in (1, 2):
                deep_values: List[float] = []
                lmmse_values: List[float] = []
                for sample_index in range(samples):
                    params = sample_paper_dataset_parameters(
                        config,
                        split="validation",
                        index=sample_index,
                        seed=seed,
                        snr_db=float(snr),
                        pilot_count=pilot_count,
                    )
                    iteration = paper_dataset_iteration(config, split="validation", index=sample_index)
                    deep_values.append(
                        bridge.evaluate_pytorch_deeprx(
                            params,
                            model_path=checkpoint_path,
                            iteration=iteration,
                            n_frames=n_frames,
                        )
                    )
                    lmmse_values.append(
                        bridge.evaluate_practical_lmmse(
                            params,
                            iteration=iteration,
                            n_frames=n_frames,
                        )
                    )
                    known_channel_values.append(
                        bridge.evaluate_known_channel_lmmse(
                            params,
                            iteration=iteration,
                            n_frames=n_frames,
                        )
                    )
                suffix = "1_pilot" if pilot_count == 1 else "2_pilots"
                metrics["curves"][f"deeprx_{suffix}"].append(_mean(deep_values))
                metrics["curves"][f"lmmse_{suffix}"].append(_mean(lmmse_values))
            metrics["curves"]["lmmse_known_channel"].append(_mean(known_channel_values))
            _write_json_atomic(
                progress_path,
                {
                    "signature": signature,
                    "completed_snr_count": snr_index + 1,
                    "metrics": metrics,
                },
            )

    metrics_path = output_dir / "figure6a_metrics.json"
    _write_json_atomic(metrics_path, metrics)
    plot_figure6a(metrics, output_dir / "figure6a_uncoded_ber.png")
    progress_path.unlink(missing_ok=True)
    return metrics


def initialize_figure6a_metrics(
    *,
    checkpoint_path: Path,
    snr_values: Iterable[float],
    samples_per_point: int,
    n_frames: int,
) -> Dict:
    return {
        "mode": "paper_figure6a_official_matlab",
        "checkpoint": str(checkpoint_path),
        "snr_db": list(snr_values),
        "samples_per_point": samples_per_point,
        "n_frames": n_frames,
        "curves": {
            "deeprx_1_pilot": [],
            "deeprx_2_pilots": [],
            "lmmse_1_pilot": [],
            "lmmse_2_pilots": [],
            "lmmse_known_channel": [],
        },
    }


def plot_figure6a(metrics: Dict, path: Path) -> None:
    snr = metrics["snr_db"]
    curves = metrics["curves"]
    plt.figure(figsize=(7.0, 5.0), dpi=160)
    plt.semilogy(snr, curves["deeprx_1_pilot"], "-o", color="blue", label="DeepRx, 1 pilot")
    plt.semilogy(snr, curves["deeprx_2_pilots"], "--D", color="blue", label="DeepRx, 2 pilots")
    plt.semilogy(snr, curves["lmmse_1_pilot"], "-s", color="red", label="LMMSE, 1 pilot")
    plt.semilogy(snr, curves["lmmse_2_pilots"], "--^", color="red", label="LMMSE, 2 pilots")
    plt.semilogy(snr, curves["lmmse_known_channel"], ":X", color="green", label="LMMSE, known channel")
    plt.xlabel("SINR (dB)")
    plt.ylabel("Uncoded BER")
    plt.ylim(1e-4, 1.0)
    if min(snr) == max(snr):
        plt.xlim(min(snr) - 0.5, max(snr) + 0.5)
    else:
        plt.xlim(min(snr), max(snr))
    plt.grid(True, which="both", alpha=0.45)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / max(len(values), 1))


def _write_json_atomic(path: Path, payload: Dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
