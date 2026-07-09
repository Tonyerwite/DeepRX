from __future__ import annotations

import json
import random
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
    sample_official_parameters,
)


def run_figure6a_reproduction(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    snr_points: Iterable[float] | None = None,
    samples_per_point: int = 1,
    n_frames: int = 1,
    seed: int = 2026,
) -> Dict:
    config = PaperFigure6Config()
    snr_values = list(config.figure6_sinr_points_db if snr_points is None else snr_points)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    metrics = {
        "mode": "paper_figure6a_official_matlab",
        "checkpoint": str(checkpoint_path),
        "snr_db": snr_values,
        "samples_per_point": samples_per_point,
        "n_frames": n_frames,
        "curves": {
            "deeprx_1_pilot": [],
            "deeprx_2_pilots": [],
            "lmmse_1_pilot": [],
            "lmmse_2_pilots": [],
        },
    }

    with MatlabDeepRxBridge(load_matlab_bridge_paths()) as bridge:
        iteration = 30_001
        for snr in snr_values:
            for pilot_count in (1, 2):
                deep_values: List[float] = []
                lmmse_values: List[float] = []
                for _ in range(samples_per_point):
                    params = sample_official_parameters(config, rng, mode="validate", snr_db=float(snr), pilot_count=pilot_count)
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
                    iteration += 1
                suffix = "1_pilot" if pilot_count == 1 else "2_pilots"
                metrics["curves"][f"deeprx_{suffix}"].append(_mean(deep_values))
                metrics["curves"][f"lmmse_{suffix}"].append(_mean(lmmse_values))

    metrics_path = output_dir / "figure6a_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_figure6a(metrics, output_dir / "figure6a_uncoded_ber.png")
    return metrics


def plot_figure6a(metrics: Dict, path: Path) -> None:
    snr = metrics["snr_db"]
    curves = metrics["curves"]
    plt.figure(figsize=(7.0, 5.0), dpi=160)
    plt.semilogy(snr, curves["deeprx_1_pilot"], "-o", color="blue", label="DeepRx, 1 pilot")
    plt.semilogy(snr, curves["deeprx_2_pilots"], "--D", color="blue", label="DeepRx, 2 pilots")
    plt.semilogy(snr, curves["lmmse_1_pilot"], "-s", color="red", label="LMMSE, 1 pilot")
    plt.semilogy(snr, curves["lmmse_2_pilots"], "--^", color="red", label="LMMSE, 2 pilots")
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
