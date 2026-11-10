"""Tanin -- real-time guitar amp, pedalboard, looper and recorder.

The ASIO switch is set here, at the top of the package, because sounddevice
chooses which PortAudio build to load at *import* time and never revisits it.
Setting it inside engine.py worked only as long as nothing imported
sounddevice first; when something did, ASIO silently vanished from the device
list and the app quietly fell back to a 45 ms path. Setting it at package
import removes that ordering trap.
"""

import os

os.environ.setdefault("SD_ENABLE_ASIO", "1")
