"""Thin BrainFlow adapter for NeuroPawn Knight IMU streaming."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from .config import AppConfig
from .ports import resolve_serial_port
from .rows import BOARD_ID_VALUE, NUM_ROWS, board_descriptor_contract, validate_batch

LogFn = Callable[[str], None]


class BoardConnectionError(RuntimeError):
    pass


class KnightBrainFlowAdapter:
    """Owns BrainFlow session lifecycle for the Knight IMU board."""

    def __init__(self, config: AppConfig, log: LogFn | None = None) -> None:
        self.config = config
        self.log = log or (lambda message: None)
        self.port = ""
        self.board: Any | None = None
        self.board_descr: dict[str, Any] = {}
        self.brainflow_version = ""
        self.is_streaming = False

    def connect(self) -> None:
        try:
            from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        except Exception as exc:  # pragma: no cover - exercised when dependency is missing
            raise BoardConnectionError(
                "BrainFlow is not installed. Run `uv sync` before connecting to hardware."
            ) from exc

        resolution = resolve_serial_port(self.config.board.serial_port)
        self.port = resolution.port
        self.log(resolution.message)

        params = BrainFlowInputParams()
        params.serial_port = self.port
        params.other_info = json.dumps({"gain": self.config.board.gain})

        try:
            BoardShim.disable_board_logger()
            self.brainflow_version = BoardShim.get_version()
            self.board_descr = BoardShim.get_board_descr(BOARD_ID_VALUE)
            self._validate_descriptor(self.board_descr)
            self.board = BoardShim(BoardIds.NEUROPAWN_KNIGHT_BOARD_IMU.value, params)
            self.board.prepare_session()
            self.log(f"Prepared BrainFlow session on {self.port}")
            self.board.start_stream(self.config.board.buffer_size)
            self.is_streaming = True
            self.log("BrainFlow stream started")
            if self.config.board.configure_channels:
                self._configure_exg_channels()
        except Exception as exc:
            self.close()
            raise BoardConnectionError(str(exc)) from exc

    def _validate_descriptor(self, descriptor: dict[str, Any]) -> None:
        contract = board_descriptor_contract()
        if int(descriptor.get("num_rows", -1)) != NUM_ROWS:
            raise BoardConnectionError(
                f"BrainFlow Knight IMU descriptor has {descriptor.get('num_rows')} rows; expected {NUM_ROWS}"
            )
        for key in ("package_num_channel", "timestamp_channel", "marker_channel", "sampling_rate"):
            if int(descriptor.get(key, -1)) != int(contract[key]):
                raise BoardConnectionError(
                    f"BrainFlow descriptor mismatch for {key}: {descriptor.get(key)} != {contract[key]}"
                )

    def _configure_exg_channels(self) -> None:
        if self.board is None:
            raise BoardConnectionError("cannot configure channels before session exists")
        if self.config.board.stream_settle_seconds > 0:
            self.log(
                f"Waiting {self.config.board.stream_settle_seconds:.1f}s before channel configuration"
            )
            time.sleep(self.config.board.stream_settle_seconds)
        active = set(self.config.board.active_exg_channels)
        for channel in sorted(active):
            self._config_board(f"chon_{channel}_{self.config.board.gain}")
            self._config_board(f"rldadd_{channel}")
        for channel in range(1, 9):
            if channel in active:
                continue
            self._config_board(f"choff_{channel}")

    def _config_board(self, command: str) -> None:
        if self.board is None:
            raise BoardConnectionError("cannot configure channels before session exists")
        response = self.board.config_board(command)
        self.log(f"config_board({command}) -> {response!r}")
        if self.config.board.config_command_delay_seconds > 0:
            time.sleep(self.config.board.config_command_delay_seconds)

    def drain(self) -> np.ndarray:
        if self.board is None:
            return np.zeros((NUM_ROWS, 0), dtype=float)
        data = self.board.get_board_data()
        return validate_batch(data)

    def current(self, samples: int = 250) -> np.ndarray:
        if self.board is None:
            return np.zeros((NUM_ROWS, 0), dtype=float)
        data = self.board.get_current_board_data(samples)
        return validate_batch(data)

    def buffer_count(self) -> int:
        if self.board is None:
            return 0
        return int(self.board.get_board_data_count())

    def close(self) -> None:
        if self.board is None:
            return
        try:
            if self.is_streaming:
                self.board.stop_stream()
                self.log("BrainFlow stream stopped")
        except Exception as exc:
            self.log(f"stop_stream failed: {exc}")
        finally:
            self.is_streaming = False
        try:
            self.board.release_session()
            self.log("BrainFlow session released")
        except Exception as exc:
            self.log(f"release_session failed: {exc}")
        finally:
            self.board = None
