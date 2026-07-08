import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deeprx.experiments import run_quick_reproduction


def main():
    parser = argparse.ArgumentParser(description="Run a tiny DeepRx reproduction smoke test.")
    parser.add_argument("--output-dir", default="outputs/quick_reproduction")
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--snr-start", type=float, default=0.0)
    parser.add_argument("--snr-stop", type=float, default=12.0)
    parser.add_argument("--snr-step", type=float, default=2.0)
    args = parser.parse_args()
    metrics = run_quick_reproduction(
        output_dir=args.output_dir,
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        device=args.device,
        snr_start=args.snr_start,
        snr_stop=args.snr_stop,
        snr_step=args.snr_step,
    )
    print(f"Wrote metrics and figures to {args.output_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
