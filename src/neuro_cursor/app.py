"""Application entrypoint."""

from __future__ import annotations

import argparse
import sys

from .config import DEFAULT_CONFIG_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeuroPawn Knight IMU diagnostics GUI")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config")
    parser.add_argument("--smoke", action="store_true", help="Run the CLI smoke test instead of the GUI")
    parser.add_argument("--seconds", type=float, default=6.0, help="Smoke-test duration")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        from .smoke import main as smoke_main

        return smoke_main(["--config", args.config, "--seconds", str(args.seconds)])

    try:
        from .gui import run_gui
    except Exception as exc:
        print(f"GUI dependencies are unavailable: {exc}", file=sys.stderr)
        print("Run `uv sync` and retry, or use `uv run neuro-smoke` for CLI diagnostics.", file=sys.stderr)
        return 2
    return run_gui(args.config)


if __name__ == "__main__":
    raise SystemExit(main())

