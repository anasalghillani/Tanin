"""Looper.

Single loop, unlimited overdubs, one level of undo -- the layout of every
hardware looper, because that is what fingers already know.

Buffers are allocated once at ``prepare()``. Anything that would touch tens of
megabytes (clearing, snapshotting for undo) happens on the *control* thread,
never in the callback: a 45 MB memset is ~10 ms, which is several blocks' worth
of deadline and would be heard as a dropout.
"""

from __future__ import annotations

import threading

import numpy as np

EMPTY = "empty"
RECORDING = "recording"
PLAYING = "playing"
OVERDUBBING = "overdubbing"
STOPPED = "stopped"

_XFADE = 192  # ~4 ms at 48 k, applied at the loop seam to kill the click


class Looper:
    def __init__(self, max_seconds: float = 150.0):
        self.max_seconds = max_seconds
        self.state = EMPTY
        self.level = 1.0
        self.speed = 1.0          # 1.0 normal, 0.5 octave down / half speed
        self.reverse = False
        self.feedback = 1.0       # <1 makes older layers fade as you overdub

        self.sr = 48000.0
        self.capacity = 0
        self.length = 0
        self.pos = 0.0
        self._buf = np.zeros((2, 0), dtype=np.float32)
        self._undo = np.zeros((2, 0), dtype=np.float32)
        self._undo_len = 0
        self._has_undo = False
        self._lock = threading.Lock()   # control thread only, never in callback

    # -- lifecycle ---------------------------------------------------------

    def prepare(self, sr: float, max_block: int) -> None:
        self.sr = float(sr)
        self.capacity = int(sr * self.max_seconds)
        self._buf = np.zeros((2, self.capacity), dtype=np.float32)
        self._undo = np.zeros((2, self.capacity), dtype=np.float32)
        self._idx = np.arange(max_block, dtype=np.float64)
        self.length = 0
        self.pos = 0.0
        self.state = EMPTY
        self._has_undo = False

    # -- control-thread actions -------------------------------------------

    def toggle(self) -> str:
        """The one-button workflow: rec -> play -> overdub -> play -> ..."""
        with self._lock:
            if self.state == EMPTY:
                self.length = 0
                self.pos = 0.0
                self.state = RECORDING
            elif self.state == RECORDING:
                self._close_loop()
                self.state = PLAYING
            elif self.state == PLAYING:
                self._snapshot_for_undo()
                self.state = OVERDUBBING
            elif self.state == OVERDUBBING:
                self.state = PLAYING
            elif self.state == STOPPED:
                self.state = PLAYING
        return self.state

    def record(self) -> str:
        with self._lock:
            if self.state in (EMPTY, STOPPED) and self.length == 0:
                self.length = 0
                self.pos = 0.0
                self.state = RECORDING
            elif self.length > 0:
                self._snapshot_for_undo()
                self.state = OVERDUBBING
        return self.state

    def play(self) -> str:
        with self._lock:
            if self.state == RECORDING:
                self._close_loop()
            if self.length > 0:
                self.state = PLAYING
        return self.state

    def stop(self) -> str:
        with self._lock:
            if self.state == RECORDING:
                self._close_loop()
            self.state = STOPPED if self.length else EMPTY
            self.pos = 0.0
        return self.state

    def clear(self) -> str:
        with self._lock:
            self.state = EMPTY
            self.length = 0
            self.pos = 0.0
            self._has_undo = False
            # No memset: the first record pass overwrites rather than sums.
        return self.state

    def undo(self) -> bool:
        with self._lock:
            if not self._has_undo or self._undo_len == 0:
                return False
            n = self._undo_len
            self._buf[:, :n], self._undo[:, :n] = (
                self._undo[:, :n].copy(), self._buf[:, :n].copy())
            self.length = n
            if self.state == OVERDUBBING:
                self.state = PLAYING
            return True

    def _close_loop(self) -> None:
        self.length = int(self.pos)
        self.pos = 0.0
        if self.length > _XFADE * 2:
            # Blend the tail into the head so the seam is continuous.
            fade = np.linspace(0.0, 1.0, _XFADE, dtype=np.float32)
            head = self._buf[:, :_XFADE]
            tail = self._buf[:, self.length - _XFADE:self.length]
            head *= fade
            head += tail * (1.0 - fade)
            self.length -= _XFADE
        elif self.length < 64:
            self.length = 0
            self.state = EMPTY

    def _snapshot_for_undo(self) -> None:
        if self.length <= 0:
            return
        self._undo[:, :self.length] = self._buf[:, :self.length]
        self._undo_len = self.length
        self._has_undo = True

    # -- audio thread ------------------------------------------------------

    def process(self, wet: np.ndarray, n: int) -> np.ndarray:
        """Record `wet` if armed and add loop playback. Returns the sum.

        Never allocates beyond one small block, never blocks -- the control
        lock is deliberately not taken here.
        """
        state = self.state
        if state == EMPTY:
            return wet

        if state == RECORDING:
            p = int(self.pos)
            k = min(n, self.capacity - p)
            if k > 0:
                self._buf[:, p:p + k] = wet[:, :k]
                self.pos = p + k
            if k < n:  # hit the ceiling -- close it rather than wrap
                self._close_loop()
                self.state = PLAYING
            return wet

        if self.length <= 0:
            return wet

        if state == STOPPED:
            return wet

        length = self.length
        speed = self.speed if self.speed > 0.01 else 1.0
        step = -speed if self.reverse else speed

        pos = self.pos + self._idx[:n] * step
        pos = np.mod(pos, length)
        i0 = pos.astype(np.int64)
        frac = pos - i0
        i1 = (i0 + 1) % length

        loop = np.empty((2, n), dtype=np.float64)
        for c in range(2):
            row = self._buf[c]
            a = row[i0]
            b = row[i1]
            loop[c] = a + (b - a) * frac

        if state == OVERDUBBING and not self.reverse and abs(speed - 1.0) < 1e-6:
            p = int(self.pos) % length
            k = min(n, length - p)
            if self.feedback < 0.999:
                self._buf[:, p:p + k] *= self.feedback
            self._buf[:, p:p + k] += wet[:, :k].astype(np.float32)
            if k < n:
                rest = n - k
                if self.feedback < 0.999:
                    self._buf[:, :rest] *= self.feedback
                self._buf[:, :rest] += wet[:, k:].astype(np.float32)

        self.pos = (self.pos + n * step) % length
        return wet + loop * self.level

    # -- telemetry ---------------------------------------------------------

    @property
    def length_seconds(self) -> float:
        return self.length / self.sr if self.sr else 0.0

    @property
    def position_fraction(self) -> float:
        return (self.pos / self.length) if self.length else 0.0

    @property
    def has_undo(self) -> bool:
        return self._has_undo
