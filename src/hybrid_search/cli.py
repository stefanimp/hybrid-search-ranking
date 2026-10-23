"""Explicit paths, input checks and a lightweight runnable ranking demonstration."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid search: LambdaRank + reciprocal rank fusion"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="Run a synthetic RRF example without the tournament data")
    for command in ["check", "run"]:
        p = sub.add_parser(command)
        p.add_argument("--data-dir", type=Path, required=True)
        p.add_argument("--cache-dir", type=Path, required=True)
        if command == "run":
            p.add_argument("--output-dir", type=Path, required=True)
            p.add_argument(
                "--include-variants",
                action="store_true",
                help="Also export the three historical fusion variants",
            )
    args = parser.parse_args(argv)
    if args.command == "demo":
        from .ranking import rrf

        rankings = [
            (1.0, {"example": ["passage-a", "passage-b"]}, 2),
            (1.0, {"example": ["passage-b", "passage-c"]}, 2),
        ]
        print(json.dumps({"synthetic": True, "rankings": rrf(rankings)}, indent=2))
        return 0
    from .io import check_inputs

    try:
        counts = check_inputs(args.data_dir, args.cache_dir)
        if args.command == "check":
            print(
                json.dumps(
                    {"status": "valid", "rows": counts, "cache_content_checked": False}, indent=2
                )
            )
            return 0
        if args.output_dir.exists() and (
            not args.output_dir.is_dir() or any(args.output_dir.iterdir())
        ):
            raise ValueError("Output directory must be new or empty to preserve previous runs.")
        from .pipeline import run

        logging.basicConfig(level=logging.INFO, format="%(message)s")
        run(args.data_dir, args.cache_dir, args.output_dir, include_variants=args.include_variants)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0
