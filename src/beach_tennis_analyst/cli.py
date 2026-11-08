from __future__ import annotations

import argparse
import json
from pathlib import Path

from beach_tennis_analyst.pipeline.session import BeachTennisAnalysisPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Beach Tennis Analyst M1 pipeline")
    parser.add_argument("video", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--progress-json",
        action="store_true",
        help="Emit machine-readable progress lines for launchers/orchestrators.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    def emit_progress(payload: dict[str, object]) -> None:
        if args.progress_json:
            print(
                "PROGRESS_JSON " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                flush=True,
            )

    outputs = BeachTennisAnalysisPipeline().run(
        video_path=args.video,
        calibration_path=args.calibration,
        output_dir=args.output,
        progress_callback=emit_progress if args.progress_json else None,
    )
    print(f"Processed {outputs.metadata.frame_count} frames at {outputs.metadata.fps:.3f} FPS")
    print(f"Outputs written to {outputs.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
