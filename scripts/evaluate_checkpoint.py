import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deeprx.experiments import evaluate_checkpoint_snr_grid


def _snr_points(start: float, stop: float, step: float):
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(float(round(current, 10)))
        current += step
    return values


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DeepRx checkpoint over an SNR grid.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/checkpoint_eval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--snr-start", type=float, default=0.0)
    parser.add_argument("--snr-stop", type=float, default=12.0)
    parser.add_argument("--snr-step", type=float, default=2.0)
    parser.add_argument("--eval-samples", type=int, default=20)
    parser.add_argument("--modulation", default="16QAM")
    parser.add_argument("--max-bits", type=int, default=4)
    parser.add_argument("--n-subcarriers", type=int, default=312)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--cp-length", type=int, default=36)
    parser.add_argument("--doppler-hz", type=float, default=250.0)
    parser.add_argument("--channel-profile", default="TDL-E")
    parser.add_argument("--pilot-config", default="2_pilots_left")
    args = parser.parse_args()

    results = evaluate_checkpoint_snr_grid(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        snr_points=_snr_points(args.snr_start, args.snr_stop, args.snr_step),
        eval_samples=args.eval_samples,
        device=args.device,
        modulation=args.modulation,
        max_bits_per_symbol=args.max_bits,
        n_subcarriers=args.n_subcarriers,
        n_fft=args.n_fft,
        cp_length=args.cp_length,
        doppler_hz=args.doppler_hz,
        channel_profile=args.channel_profile,
        pilot_config=args.pilot_config,
    )
    print(f"Wrote checkpoint evaluation to {args.output_dir}")
    print(results)


if __name__ == "__main__":
    main()
