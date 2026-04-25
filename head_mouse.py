"""
Head Mouse - IMU-based cursor control for people who cannot use their hands.
Tilt your head left/right to move the mouse horizontally,
tilt forward/back to move vertically.
"""

import time
import pyautogui
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# ─── CONFIG (change these) ────────────────────────────────────────────────────
SERIAL_PORT = "COM3"   # <-- check your EXG Visualizer dropdown and put your port here
SENSITIVITY  = 20      # higher = faster cursor
DEAD_ZONE    = 0.25    # ignore tiny tilts (reduces drift)
SAMPLE_RATE  = 10      # how many recent samples to average
# ─────────────────────────────────────────────────────────────────────────────

# IMU uses its own dedicated board ID — separate from the EEG board
BOARD_ID = BoardIds.NEUROPAWN_KNIGHT_BOARD_IMU  # 66

# Will be set automatically from BrainFlow after connecting
ACCEL_X = None
ACCEL_Y = None


def connect_board():
    global ACCEL_X, ACCEL_Y

    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT

    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    board.start_stream()

    # Look up correct accelerometer channel indices for this board
    accel = BoardShim.get_accel_channels(BOARD_ID)
    ACCEL_X = accel[0]  # X axis — left/right tilt
    ACCEL_Y = accel[1]  # Y axis — forward/back tilt
    print(f"Accel channels: X={ACCEL_X}, Y={ACCEL_Y}")

    print(f"Connected on {SERIAL_PORT}. Move mouse to TOP-LEFT corner to emergency-stop.")
    time.sleep(2)
    return board


def run(board):
    pyautogui.FAILSAFE = True  # moving mouse to corner raises exception → clean exit

    while True:
        data = board.get_current_board_data(SAMPLE_RATE)

        if data.shape[1] < 1:
            time.sleep(0.02)
            continue

        ax = float(np.mean(data[ACCEL_X]))  # tilt left/right
        ay = float(np.mean(data[ACCEL_Y]))  # tilt forward/back

        # dead zone — ignore small noise
        dx = ax * SENSITIVITY if abs(ax) > DEAD_ZONE else 0.0
        dy = ay * SENSITIVITY if abs(ay) > DEAD_ZONE else 0.0

        # clamp to ±100 pixels per frame so we never overflow
        dx = max(-100, min(100, int(dx)))
        dy = max(-100, min(100, int(dy)))

        if dx != 0 or dy != 0:
            pyautogui.moveRel(dx, -dy, duration=0)

        time.sleep(0.03)  # ~30 fps


def main():
    board = connect_board()
    try:
        run(board)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except pyautogui.FailSafeException:
        print("\nEmergency stop triggered (mouse moved to corner).")
    finally:
        board.stop_stream()
        board.release_session()
        print("Board disconnected.")


if __name__ == "__main__":
    main()
