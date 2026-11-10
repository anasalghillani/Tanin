"""Tanin -- guitar amp, pedalboard, looper and recorder.

Usage:
    python run.py            launch the app
    python run.py --devices  list audio devices and exit
"""

import os
import sys

# Must precede any sounddevice import; see amprack/__init__.py.
os.environ.setdefault("SD_ENABLE_ASIO", "1")


def main() -> int:
    if "--devices" in sys.argv:
        from amprack.engine import list_devices
        for kind in ("input", "output"):
            print(f"\n{kind.upper()}")
            for d in list_devices(kind):
                print(f"  {d['index']:3d}  [{d['api']:<18}] {d['channels']}ch "
                      f"{d['default_sr']:>6} Hz   {d['name']}")
        return 0
    from amprack.ui.main import run
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
