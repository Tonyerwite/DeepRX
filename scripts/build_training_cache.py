import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deeprx.matlab_bridge import MatlabDeepRxBridge, PaperFigure6Config, paper_dataset_split_frame_counts
from deeprx.training_cache import build_paper_training_cache


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build the fixed MATLAB-generated DeepRx train-frame cache.")
    parser.add_argument("--cache-dir", default="data/paper_train_cache")
    parser.add_argument("--frame-count", type=int, default=0, help="Train frames to cache. Default 0 means the full paper train split.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main():
    args = build_arg_parser().parse_args()
    config = PaperFigure6Config()
    train_frames, _ = paper_dataset_split_frame_counts(config)
    frame_count = train_frames if args.frame_count == 0 else args.frame_count

    def report(current: int, total: int) -> None:
        print(f"cached_frames={current}/{total}", flush=True)

    with MatlabDeepRxBridge() as bridge:
        build_paper_training_cache(
            bridge,
            args.cache_dir,
            config=config,
            seed=args.seed,
            frame_count=frame_count,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
            progress_callback=report,
        )
    print(f"Built training cache at {args.cache_dir} with {frame_count} frames")


if __name__ == "__main__":
    main()
