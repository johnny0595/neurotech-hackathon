"""Row-based hardware smoke test for the Knight IMU board."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .brainflow_adapter import BoardConnectionError, KnightBrainFlowAdapter
from .config import DEFAULT_CONFIG_PATH, load_config
from .ports import list_usbserial_ports
from .rows import (
    ACCEL_ROWS,
    EXG_ROWS,
    GYRO_ROWS,
    MAG_ROWS,
    NUM_ROWS,
    PACKAGE_ROW,
    ROW_LABELS,
    TIMESTAMP_ROW,
    row_stats,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config")
    parser.add_argument("--seconds", type=float, default=6.0, help="Streaming duration")
    parser.add_argument("--warmup-timeout", type=float, default=5.0, help="Seconds to wait for first packets")
    parser.add_argument(
        "--no-configure-channels",
        action="store_true",
        help="Skip EXG chon/choff/rldadd commands and read the stream as-is",
    )
    parser.add_argument(
        "--disable-exg",
        action="store_true",
        help="Send choff_1..choff_8 before collecting, useful for IMU-only cursor checks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.no_configure_channels:
        config.board.configure_channels = False
    if args.disable_exg:
        config.board.configure_channels = True
        config.board.active_exg_channels = []
    logs: list[str] = []

    def log(message: str) -> None:
        logs.append(message)
        print(f"[brainflow] {message}")

    adapter = KnightBrainFlowAdapter(config, log=log)
    batches: list[np.ndarray] = []
    started = 0.0
    ended = 0.0
    try:
        adapter.connect()
        print(f"Board descriptor: {adapter.board_descr}")
        print(f"BrainFlow version: {adapter.brainflow_version}")
        print(f"Waiting for first packets, then collecting for {args.seconds:.1f}s...")
        warmup_started = time.monotonic()
        while adapter.buffer_count() == 0 and time.monotonic() - warmup_started < args.warmup_timeout:
            time.sleep(0.05)
        if adapter.buffer_count() == 0:
            raise BoardConnectionError(
                f"stream opened, but no packets arrived within {args.warmup_timeout:.1f}s"
            )
        warmup = adapter.drain()
        print(f"Warmup packets discarded: {warmup.shape[1]}")
        started = time.monotonic()
        while time.monotonic() - started < args.seconds:
            batch = adapter.drain()
            if batch.shape[1] > 0:
                batches.append(batch)
            time.sleep(0.05)
        ended = time.monotonic()
    except BoardConnectionError as exc:
        print(f"Connection failed: {exc}", file=sys.stderr)
        return 2
    finally:
        adapter.close()

    if not batches:
        print("No packages were received from the Knight IMU stream.", file=sys.stderr)
        print(f"Visible serial ports: {list_usbserial_ports()}", file=sys.stderr)
        print(
            "The port opened successfully, but the device did not emit BrainFlow frames. "
            "Check board power/firmware mode and whether another app has the serial port open.",
            file=sys.stderr,
        )
        return 3

    data = np.concatenate(batches, axis=1)
    elapsed = max((ended or time.monotonic()) - started, 0.001) if started else 0.001
    wall_rate = data.shape[1] / elapsed
    print(f"\nReceived shape: rows={data.shape[0]} samples={data.shape[1]}")
    print(f"Wall-window package rate: {wall_rate:.1f} Hz")
    if data.shape[0] != NUM_ROWS:
        print(f"Expected {NUM_ROWS} rows, got {data.shape[0]}", file=sys.stderr)
        return 4

    counters = data[PACKAGE_ROW].astype(int)
    if counters.size > 1:
        deltas = (np.diff(counters) % 256).astype(int)
        missed = int(np.sum(deltas != 1))
        print(f"Counter gaps: {missed}")

    timestamp_span = float(data[TIMESTAMP_ROW, -1] - data[TIMESTAMP_ROW, 0])
    if timestamp_span > 0:
        print(f"Timestamp-derived rate: {(data.shape[1] - 1) / timestamp_span:.1f} Hz")

    print("\nKey row groups:")
    print(f"  EXG rows:   {EXG_ROWS}")
    print(f"  Accel rows: {ACCEL_ROWS}")
    print(f"  Gyro rows:  {GYRO_ROWS}")
    print(f"  Mag rows:   {MAG_ROWS}")

    print("\nRow stats:")
    for stat in row_stats(data):
        print(
            f"  {stat.row:02d} {ROW_LABELS[stat.row]:>16}: "
            f"latest={stat.latest:>12.4f} mean={stat.mean:>12.4f} spread={stat.spread:>10.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
