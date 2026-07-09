import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deeprx.matlab_bridge import load_matlab_bridge_paths
from deeprx.official_experiments import run_figure6a_reproduction


def build_arg_parser():
    paths = load_matlab_bridge_paths(ROOT)
    parser = argparse.ArgumentParser(description="Reproduce DeepRx paper Fig. 6(a) uncoded BER with official MATLAB PUSCH simulation.")
    parser.add_argument("--checkpoint", default=str(paths.default_checkpoint))
    parser.add_argument("--output-dir", default="outputs/figure6a")
    parser.add_argument("--samples-per-point", type=int, default=500)
    parser.add_argument("--n-frames", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--snr-points", default="", help="Comma-separated SNR/SINR points. Default: paper 0,3,...,21 dB.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    snr_points = None
    if args.snr_points:
        snr_points = [float(item.strip()) for item in args.snr_points.split(",") if item.strip()]

    metrics = run_figure6a_reproduction(
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        snr_points=snr_points,
        samples_per_point=args.samples_per_point,
        n_frames=args.n_frames,
        seed=args.seed,
    )
    print(f"Wrote {Path(args.output_dir) / 'figure6a_metrics.json'}")
    print(f"Wrote {Path(args.output_dir) / 'figure6a_uncoded_ber.png'}")
    print(metrics)


if __name__ == "__main__":
    main()
