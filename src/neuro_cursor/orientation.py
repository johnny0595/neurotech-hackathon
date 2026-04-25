"""Orientation estimation for Knight IMU diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, asin, cos, degrees, radians, sin

import numpy as np

from .config import ImuConfig
from .rows import accel, gyro, mag


@dataclass(frozen=True)
class OrientationSample:
    quaternion: tuple[float, float, float, float]
    roll: float
    pitch: float
    yaw: float
    raw_roll: float
    raw_pitch: float
    raw_yaw: float
    dt: float
    accel: tuple[float, float, float]
    gyro: tuple[float, float, float]
    mag: tuple[float, float, float]
    calibration_offset: tuple[float, float, float, float]


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm <= 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def quat_to_euler_degrees(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = _normalize_quat(q)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = asin(float(np.clip(sinp, -1.0, 1.0)))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = atan2(siny_cosp, cosy_cosp)
    return degrees(roll), degrees(pitch), degrees(yaw)


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    return _normalize_quat(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=float,
        )
    )


class OrientationEstimator:
    def __init__(self, config: ImuConfig) -> None:
        self.config = config
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.offset = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.last_timestamp: float | None = None
        self._madgwick = None
        self._load_madgwick()

    @property
    def backend(self) -> str:
        return "ahrs.Madgwick" if self._madgwick is not None else "accelerometer fallback"

    def _load_madgwick(self) -> None:
        try:
            from ahrs.filters import Madgwick

            self._madgwick = Madgwick(gain=self.config.beta)
        except Exception:
            self._madgwick = None

    def update_config(self, config: ImuConfig) -> None:
        self.config = config
        if self._madgwick is not None:
            self._madgwick.gain = config.beta

    def zero_level(self) -> None:
        self.offset = quat_conjugate(self.q)

    def update(self, raw: np.ndarray) -> OrientationSample:
        ts = float(raw[20])
        dt = 1.0 / 125.0
        if self.last_timestamp is not None:
            measured = ts - self.last_timestamp
            if 0.001 <= measured <= 0.200:
                dt = measured
        self.last_timestamp = ts

        acc = accel(raw)
        gyr = self._gyro_rad_s(gyro(raw))
        magnetic = mag(raw)
        self.q = self._update_quaternion(acc, gyr, magnetic, dt)

        raw_roll, raw_pitch, raw_yaw = quat_to_euler_degrees(self.q)
        corrected_q = _normalize_quat(quat_multiply(self.offset, self.q))
        roll, pitch, yaw = quat_to_euler_degrees(corrected_q)
        if self.config.swap_pitch_roll:
            roll, pitch = pitch, roll
        if self.config.invert_roll:
            roll = -roll

        return OrientationSample(
            quaternion=tuple(float(x) for x in corrected_q),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            raw_roll=float(raw_roll),
            raw_pitch=float(raw_pitch),
            raw_yaw=float(raw_yaw),
            dt=float(dt),
            accel=tuple(float(x) for x in acc),
            gyro=tuple(float(x) for x in gyr),
            mag=tuple(float(x) for x in magnetic),
            calibration_offset=tuple(float(x) for x in self.offset),
        )

    def _gyro_rad_s(self, gyr: np.ndarray) -> np.ndarray:
        units = self.config.gyro_units
        if units == "deg_s":
            return np.deg2rad(gyr)
        if units == "rad_s":
            return gyr.astype(float)
        # Heuristic for unknown firmware units: human head angular velocity in rad/s
        # is usually single digits, while deg/s commonly reaches tens or hundreds.
        if float(np.linalg.norm(gyr)) > 20.0:
            return np.deg2rad(gyr)
        return gyr.astype(float)

    def _update_quaternion(
        self, acc: np.ndarray, gyr: np.ndarray, magnetic: np.ndarray, dt: float
    ) -> np.ndarray:
        if np.linalg.norm(acc) <= 0:
            return self.q
        if self._madgwick is not None:
            try:
                if self.config.use_magnetometer and np.linalg.norm(magnetic) > 0:
                    q = self._madgwick.updateMARG(self.q, gyr=gyr, acc=acc, mag=magnetic, dt=dt)
                else:
                    q = self._madgwick.updateIMU(self.q, gyr=gyr, acc=acc, dt=dt)
                return _normalize_quat(np.asarray(q, dtype=float))
            except Exception:
                pass

        ax, ay, az = acc
        roll = atan2(ay, az)
        pitch = atan2(-ax, (ay * ay + az * az) ** 0.5)
        _, _, yaw_degrees = quat_to_euler_degrees(self.q)
        return euler_to_quat(roll, pitch, radians(yaw_degrees))

