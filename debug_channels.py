"""
Diagnostic - reads IMU data using the correct NEUROPAWN_KNIGHT_BOARD_IMU board ID.
"""

import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

SERIAL_PORT = "COM3"

board_id = BoardIds.NEUROPAWN_KNIGHT_BOARD_IMU  # 66 — IMU-specific board ID

print("=== IMU CHANNEL ASSIGNMENTS ===")
try:
    print("Accel channels:", BoardShim.get_accel_channels(board_id))
except Exception as e:
    print(f"Accel: {e}")
try:
    print("Gyro channels: ", BoardShim.get_gyro_channels(board_id))
except Exception as e:
    print(f"Gyro:  {e}")

print("\n=== LIVE IMU DATA (tilt your head) ===")

params = BrainFlowInputParams()
params.serial_port = SERIAL_PORT

board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream()
print("Streaming... tilt your head left/right and forward/back\n")
time.sleep(3)

try:
    for i in range(20):
        data = board.get_current_board_data(10)
        if data.shape[1] < 1:
            print(f"  [{i}] No data yet...")
            time.sleep(0.5)
            continue

        print(f"[sample {i}] channels={data.shape[0]}  packets={data.shape[1]}")
        for ch in range(data.shape[0]):
            mean   = float(np.mean(data[ch]))
            spread = float(np.max(data[ch]) - np.min(data[ch]))
            print(f"  ch{ch:02d}:  mean={mean:>12.4f}   spread={spread:>10.4f}")
        print("---")
        time.sleep(1)

except KeyboardInterrupt:
    pass
finally:
    board.stop_stream()
    board.release_session()
    print("Done.")
