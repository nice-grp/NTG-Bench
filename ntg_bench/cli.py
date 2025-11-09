from __future__ import annotations

import argparse
import logging

from .benchmark import BenchmarkRunner
from .config import BenchmarkConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NTG-Bench evaluation framework for synthetic traffic generators."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the benchmark configuration file (JSON).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity level for console logging.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    config = BenchmarkConfig.from_json(args.config)
    runner = BenchmarkRunner(config)
    runner.run()
    
if __name__ == "__main__":
    main()