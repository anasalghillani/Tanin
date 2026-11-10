"""Disk recorder.

The audio thread only ever copies into a preallocated ring; a background
thread does the file I/O. Writing a WAV from the callback would eventually
hit a filesystem stall and produce a dropout at exactly the moment you were
recording something worth keeping.
"""

from __future__ import annotations

import datetime
import os
import threading
import time

import numpy as np
import soundfile as sf


class Recorder:
    def __init__(self, out_dir: str, ring_seconds: float = 8.0):
        self.out_dir = out_dir
        self.ring_seconds = ring_seconds
        self.recording = False
        self.record_dry = True
        self.frames_written = 0
        self.wet_path = ""
        self.dry_path = ""
        self.last_error = ""

        self.sr = 48000.0
        self._ring = np.zeros((3, 0), dtype=np.float32)  # L, R, dry
        self._cap = 0
        self._w = 0
        self._r = 0
        self._thread = None
        self._stop = threading.Event()
        self._overruns = 0

    def prepare(self, sr: float) -> None:
        self.sr = float(sr)
        self._cap = int(sr * self.ring_seconds)
        self._ring = np.zeros((3, self._cap), dtype=np.float32)
        self._w = 0
        self._r = 0

    # -- control thread ----------------------------------------------------

    def start(self, name_hint: str = "take") -> str:
        if self.recording:
            return self.wet_path
        os.makedirs(self.out_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.wet_path = os.path.join(self.out_dir, f"{name_hint}-{stamp}.wav")
        self.dry_path = os.path.join(self.out_dir, f"{name_hint}-{stamp}-DI.wav")
        self.frames_written = 0
        self._overruns = 0
        self.last_error = ""
        self._w = 0
        self._r = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._writer, daemon=True,
                                        name="amprack-recorder")
        self.recording = True
        self._thread.start()
        return self.wet_path

    def stop(self) -> str:
        if not self.recording:
            return ""
        self.recording = False
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=5.0)
        self._thread = None
        return self.wet_path

    # -- audio thread ------------------------------------------------------

    def push(self, wet: np.ndarray, dry: np.ndarray, n: int) -> None:
        if not self.recording or self._cap == 0:
            return
        w = self._w
        end = w + n
        if end <= self._cap:
            self._ring[0, w:end] = wet[0]
            self._ring[1, w:end] = wet[1]
            self._ring[2, w:end] = dry
        else:
            first = self._cap - w
            self._ring[0, w:] = wet[0, :first]
            self._ring[1, w:] = wet[1, :first]
            self._ring[2, w:] = dry[:first]
            rest = end - self._cap
            self._ring[0, :rest] = wet[0, first:]
            self._ring[1, :rest] = wet[1, first:]
            self._ring[2, :rest] = dry[first:]
        self._w = end % self._cap

    # -- writer thread -----------------------------------------------------

    def _writer(self) -> None:
        wet_f = dry_f = None
        try:
            wet_f = sf.SoundFile(self.wet_path, mode="w", samplerate=int(self.sr),
                                 channels=2, subtype="FLOAT")
            if self.record_dry:
                dry_f = sf.SoundFile(self.dry_path, mode="w", samplerate=int(self.sr),
                                     channels=1, subtype="FLOAT")
            while True:
                w = self._w
                r = self._r
                if r == w:
                    if self._stop.is_set():
                        break
                    time.sleep(0.01)
                    continue
                if w > r:
                    chunk = self._ring[:, r:w]
                    self._r = w
                else:
                    chunk = self._ring[:, r:].copy()
                    self._r = 0
                wet_f.write(chunk[:2].T)
                if dry_f is not None:
                    dry_f.write(chunk[2])
                self.frames_written += chunk.shape[1]
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI status bar
            self.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            for f in (wet_f, dry_f):
                if f is not None:
                    try:
                        f.close()
                    except Exception:
                        pass

    @property
    def seconds_written(self) -> float:
        return self.frames_written / self.sr if self.sr else 0.0
