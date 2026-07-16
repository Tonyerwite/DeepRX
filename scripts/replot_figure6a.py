import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deeprx.official_experiments import write_paper_snr_figure6a_artifacts


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Redraw completed DeepRx Fig. 6(a) metrics on the paper whole-band SNR axis."
    )
    parser.add_argument("--metrics", required=True, help="Completed raw figure6a_metrics.json file.")
    parser.add_argument("--output-metrics", default="", help="Corrected metrics JSON output path.")
    parser.add_argument("--output-figure", default="", help="Corrected PNG output path.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    source_path = Path(args.metrics)
    output_metrics = (
        Path(args.output_metrics)
        if args.output_metrics
        else source_path.with_name("figure6a_metrics_paper_snr.json")
    )
    output_figure = (
        Path(args.output_figure)
        if args.output_figure
        else source_path.with_name("figure6a_uncoded_ber_paper_snr.png")
    )
    write_paper_snr_figure6a_artifacts(source_path, output_metrics, output_figure)
    print(f"Wrote {output_metrics}")
    print(f"Wrote {output_figure}")


if __name__ == "__main__":
    main()
