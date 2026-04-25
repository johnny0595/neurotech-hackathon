"""Pointer-control safety stub.

The diagnostics GUI owns pointer movement behind its explicit Arm Cursor control.
This legacy script stays disabled so it cannot bypass the GUI safety path.
"""

from __future__ import annotations

import sys


def main() -> int:
    print("Direct mouse control is disabled.")
    print("Run `uv run neuro-cursor` and use the GUI Arm Cursor control.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
