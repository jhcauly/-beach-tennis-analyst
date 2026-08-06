from __future__ import annotations

import argparse
from pathlib import Path

from beach_tennis_analyst.pipeline.session import BeachTennisAnalysisPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Beach Tennis Analyst M1 pipeline")
    parser.add_argument("video", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = BeachTennisAnalysisPipeline().run(
        video_path=args.video,
        calibration_path=args.calibration,
        output_dir=args.output,
    )
    print(f"Processed {outputs.metadata.frame_count} frames at {outputs.metadata.fps:.3f} FPS")
    print(f"Outputs written to {outputs.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
