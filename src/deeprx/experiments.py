import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from deeprx.data import DeepRxDataset
from deeprx.model import DeepRx, DeepRxLoss, compute_ber
from deeprx.receiver import LMMSEReceiver


def run_quick_reproduction(
    output_dir,
    train_steps: int = 3,
    eval_batches: int = 2,
    batch_size: int = 2,
    device: str = "cpu",
    snr_start: float = 0.0,
    snr_stop: float = 12.0,
    snr_step: float = 2.0,
) -> Dict[str, List[float]]:
    """Run a tiny end-to-end reproduction smoke test and write figures.

    This is intentionally small enough for laptops. Use the CLI with larger
    arguments for the paper-size 312-subcarrier run.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    model = DeepRx(n_rx_antennas=2, max_bits_per_symbol=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = DeepRxLoss()

    train_dataset = DeepRxDataset(
        n_samples=max(train_steps, 1),
        n_subcarriers=72,
        n_fft=128,
        cp_length=16,
        modulation="QPSK",
        snr_range=(8.0, 18.0),
        doppler_range=(0.0, 250.0),
        channel_profiles=("TDL-B", "TDL-C"),
        pilot_configs=("1_pilot_left", "2_pilots_left"),
        device=str(device),
    )
    train_losses = []
    for step in range(train_steps):
        sample = train_dataset[step % len(train_dataset)]
        loss = model.train_step(
            sample.inputs.unsqueeze(0).to(device),
            sample.target_bits.unsqueeze(0).to(device),
            sample.data_mask.to(device),
            sample.bit_mask.to(device),
            optimizer,
        )
        train_losses.append(loss)

    snr_points = _inclusive_range(snr_start, snr_stop, snr_step)
    doppler_points = [0.0, 250.0, 500.0]
    snr_results = _evaluate_snr_curve(model, snr_points, eval_batches, batch_size, device)
    doppler_results = _evaluate_doppler_curve(model, doppler_points, eval_batches, batch_size, device)

    metrics = {
        "mode": "smoke",
        "note": (
            "This short run verifies the pipeline and plotting. "
            "It is not a paper-level BER curve unless a trained checkpoint is used."
        ),
        "train_loss": train_losses,
        "snr_db": snr_points,
        "deeprx_ber": snr_results["deeprx"],
        "lmmse_ber": snr_results["lmmse"],
        "doppler_hz": doppler_points,
        "deeprx_ber_vs_doppler": doppler_results["deeprx"],
        "lmmse_ber_vs_doppler": doppler_results["lmmse"],
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _plot_curve(output / "ber_vs_snr.png", snr_points, snr_results, "SNR (dB)", "Uncoded BER", "DeepRx quick BER vs SNR")
    _plot_curve(
        output / "ber_vs_doppler.png",
        doppler_points,
        doppler_results,
        "Maximum Doppler shift (Hz)",
        "Uncoded BER",
        "DeepRx quick BER vs Doppler",
    )
    return metrics


def _inclusive_range(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(float(round(current, 10)))
        current += step
    return values


def _evaluate_snr_curve(model, snr_points, eval_batches, batch_size, device):
    deeprx, lmmse = [], []
    for snr in snr_points:
        d, l = _evaluate(model, eval_batches, batch_size, device, snr_range=(snr, snr), doppler_range=(100.0, 100.0))
        deeprx.append(d)
        lmmse.append(l)
    return {"deeprx": deeprx, "lmmse": lmmse}


def _evaluate_doppler_curve(model, doppler_points, eval_batches, batch_size, device):
    deeprx, lmmse = [], []
    for doppler in doppler_points:
        d, l = _evaluate(model, eval_batches, batch_size, device, snr_range=(12.0, 12.0), doppler_range=(doppler, doppler))
        deeprx.append(d)
        lmmse.append(l)
    return {"deeprx": deeprx, "lmmse": lmmse}


@torch.no_grad()
def _evaluate(model, eval_batches, batch_size, device, snr_range, doppler_range):
    dataset = DeepRxDataset(
        n_samples=eval_batches * batch_size,
        n_subcarriers=72,
        n_fft=128,
        cp_length=16,
        modulation="QPSK",
        snr_range=snr_range,
        doppler_range=doppler_range,
        channel_profiles=("TDL-E",),
        pilot_configs=("2_pilots_left",),
        device=str(device),
    )
    receiver = LMMSEReceiver("QPSK", device=str(device))
    deep_total = 0.0
    lmmse_total = 0.0
    n = 0
    model.eval()
    for idx in range(len(dataset)):
        sample = dataset[idx]
        logits = model(sample.inputs.unsqueeze(0).to(device))
        deep_total += compute_ber(logits, sample.target_bits.unsqueeze(0).to(device), sample.data_mask.to(device), sample.bit_mask.to(device))
        llrs = receiver.process(
            sample.rx_grid.unsqueeze(0).to(device),
            sample.tx_pilots.unsqueeze(0).to(device),
            sample.pilot_mask.unsqueeze(0).to(device),
        )
        lmmse_total += compute_ber(llrs, sample.target_bits.unsqueeze(0).to(device), sample.data_mask.to(device), sample.bit_mask.to(device))
        n += 1
    return deep_total / max(n, 1), lmmse_total / max(n, 1)


def _plot_curve(path: Path, x_values, results, xlabel: str, ylabel: str, title: str) -> None:
    plt.figure(figsize=(7, 5), dpi=140)
    plt.semilogy(x_values, results["deeprx"], "-o", label="DeepRx")
    plt.semilogy(x_values, results["lmmse"], "--s", label="LMMSE")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def evaluate_checkpoint_snr_grid(
    checkpoint_path,
    output_dir,
    snr_points=None,
    eval_samples: int = 20,
    device: str = "cpu",
    modulation: str = "16QAM",
    max_bits_per_symbol: int = 4,
    n_subcarriers: int = 312,
    n_fft: int = 512,
    cp_length: int = 36,
    n_symbols: int = 14,
    doppler_hz: float = 250.0,
    channel_profile: str = "TDL-E",
    pilot_config: str = "2_pilots_left",
) -> Dict[str, List[float]]:
    if snr_points is None:
        snr_points = _inclusive_range(0.0, 12.0, 2.0)
    snr_points = [float(x) for x in snr_points]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    model = DeepRx(n_rx_antennas=2, max_bits_per_symbol=max_bits_per_symbol).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    receiver = LMMSEReceiver(modulation, device=str(device), max_bits_per_symbol=max_bits_per_symbol)
    deeprx_ber = []
    lmmse_ber = []
    with torch.no_grad():
        for snr in snr_points:
            dataset = DeepRxDataset(
                n_samples=eval_samples,
                n_subcarriers=n_subcarriers,
                n_fft=n_fft,
                cp_length=cp_length,
                n_symbols=n_symbols,
                modulation=modulation,
                max_bits_per_symbol=max_bits_per_symbol,
                snr_range=(snr, snr),
                doppler_range=(doppler_hz, doppler_hz),
                channel_profiles=(channel_profile,),
                pilot_configs=(pilot_config,),
                device=str(device),
            )
            deep_total = 0.0
            lmmse_total = 0.0
            for idx in range(eval_samples):
                sample = dataset[idx]
                logits = model(sample.inputs.unsqueeze(0).to(device))
                deep_total += compute_ber(
                    logits,
                    sample.target_bits.unsqueeze(0).to(device),
                    sample.data_mask.to(device),
                    sample.bit_mask.to(device),
                )
                llrs = receiver.process(
                    sample.rx_grid.unsqueeze(0).to(device),
                    sample.tx_pilots.unsqueeze(0).to(device),
                    sample.pilot_mask.unsqueeze(0).to(device),
                )
                lmmse_total += compute_ber(
                    llrs,
                    sample.target_bits.unsqueeze(0).to(device),
                    sample.data_mask.to(device),
                    sample.bit_mask.to(device),
                )
            deeprx_ber.append(deep_total / max(eval_samples, 1))
            lmmse_ber.append(lmmse_total / max(eval_samples, 1))

    results = {
        "checkpoint": str(checkpoint_path),
        "snr_db": snr_points,
        "deeprx_ber": deeprx_ber,
        "lmmse_ber": lmmse_ber,
        "modulation": modulation,
        "max_bits_per_symbol": max_bits_per_symbol,
        "n_subcarriers": n_subcarriers,
        "channel_profile": channel_profile,
        "doppler_hz": doppler_hz,
        "pilot_config": pilot_config,
    }
    (output / "checkpoint_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot_curve(
        output / "checkpoint_ber_vs_snr.png",
        snr_points,
        {"deeprx": deeprx_ber, "lmmse": lmmse_ber},
        "SNR (dB)",
        "Uncoded BER",
        f"Checkpoint BER vs SNR ({modulation}, {channel_profile})",
    )
    return results
