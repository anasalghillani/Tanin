"""Looper.

Single loop, unlimited overdubs, one level of undo -- the layout of every
hardware looper, because that is what fingers already know.

Buffers are allocated once at ``prepare()`` and survive a stream restart, so
changing the buffer size does not throw away a take. Anything that would
touch tens of megabytes (snapshotting for undo) happens on the *control*
thread, never in the callback: a 45 MB memcpy is ~10 ms, which is several
blocks' worth of deadline and would be heard as a dropout.

State changes themselves are *posted*: the buttons queue a small closure that
the engine runs at the top of the next audio block. That way the callback
never sees a half-applied transition (a new length with an old position, a
buffer swapped mid-block). Without an engine attached (the harness), ``post``
just runs the closure immediately.

Clear and undo are pointer swaps between two equally sized buffers, so both
are O(1) in the callback, and a cleared loop stays in the spare buffer until
something new is recorded over it -- which is what makes Clear undoable.
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


def _run_now(fn) -> None:
    fn()


class Looper:
    def __init__(self, max_seconds: float = 150.0):
        self.max_seconds = max_seconds
        self.state = EMPTY
        self.level = 1.0
        self.speed = 1.0          # 1.0 normal, 0.5 octave down / half speed
        self.reverse = False
        self.feedback = 1.0       # <1 makes older layers fade as you overdub

        # The engine replaces this with its post(), so state changes land
        # between audio blocks. Standalone (harness) it runs them at once.
        self.post = _run_now

        self.sr = 0.0
        self.capacity = 0
        self.length = 0
        self.pos = 0.0
        self._buf = np.zeros((2, 0), dtype=np.float32)
        self._undo = np.zeros((2, 0), dtype=np.float32)
        self._undo_len = 0
        self._has_undo = False
        self._idx = np.zeros(0, dtype=np.float64)
        self._lock = threading.Lock()   # control thread only, never in callback

    # -- lifecycle ---------------------------------------------------------

    def prepare(self, sr: float, max_block: int) -> None:
        """Size for a stream. Called with the stream stopped.

        A restart at the same sample rate keeps the loop, its undo and its
        settings; only the per-block index grows if the block did. A new rate
        would replay the loop at the wrong pitch, so that one starts fresh.
        """
        if len(self._idx) < max_block:
            self._idx = np.arange(max_block, dtype=np.float64)
        if float(sr) == self.sr and self.capacity == int(sr * self.max_seconds):
            if self.state == RECORDING:
                self._close_loop()
            if self.length <= 0:
                self.state = EMPTY
            elif self.state != STOPPED:
                self.state = PLAYING      # an interrupted overdub just plays
            self.pos = 0.0
            return
        self.sr = float(sr)
        self.capacity = int(sr * self.max_seconds)
        self._buf = np.zeros((2, self.capacity), dtype=np.float32)
        self._undo = np.zeros((2, self.capacity), dtype=np.float32)
        self.length = 0
        self.pos = 0.0
        self.state = EMPTY
        self._undo_len = 0
        self._has_undo = False

    # -- control-thread actions -------------------------------------------
    #
    # Each one decides what to do when its closure *runs*, not when the
    # button was pressed, so two quick presses inside one block still step
    # through the states in order. Return values describe the state as far
    # as it is known right now; with an engine attached the change itself
    # lands a block later.

    def toggle(self) -> str:
        """The one-button workflow: rec -> play -> overdub -> play -> ..."""
        snap = self._prepare_snapshot()

        def apply():
            s = self.state
            if s == EMPTY:
                self._start_recording()
            elif s == RECORDING:
                self._close_loop()
                if self.length:
                    self.state = PLAYING
            elif s == PLAYING:
                self._begin_overdub(snap)
            elif s in (OVERDUBBING, STOPPED):
                self.state = PLAYING
        self.post(apply)
        return self.state

    def record(self) -> str:
        snap = self._prepare_snapshot()

        def apply():
            if self.length == 0 and self.state in (EMPTY, STOPPED):
                self._start_recording()
            elif self.length > 0 and self.state in (PLAYING, STOPPED):
                self._begin_overdub(snap)
        self.post(apply)
        return self.state

    def play(self) -> str:
        def apply():
            if self.state == RECORDING:
                self._close_loop()
            if self.length > 0:
                self.state = PLAYING
        self.post(apply)
        return self.state

    def stop(self) -> str:
        def apply():
            if self.state == RECORDING:
                self._close_loop()
            self.state = STOPPED if self.length else EMPTY
            self.pos = 0.0
        self.post(apply)
        return self.state

    def clear(self) -> str:
        """Empty the looper, keeping the old loop as the undo step."""
        def apply():
            if self.state == RECORDING:
                self._close_loop()
            if self.length > 0:
                # The loop moves to the spare buffer and the next recording
                # goes into the other one, so Undo can still bring it back.
                # No memset: the first record pass overwrites, not sums.
                self._buf, self._undo = self._undo, self._buf
                self._undo_len = self.length
                self._has_undo = True
            self.state = EMPTY
            self.length = 0
            self.pos = 0.0
        self.post(apply)
        return self.state

    def undo(self) -> bool:
        """Swap the loop with the undo step; pressing again redoes.

        Returns whether there was anything to undo at the time of the press.
        """
        available = self._has_undo and self._undo_len > 0

        def apply():
            if not self._has_undo or self._undo_len == 0:
                return
            if self.state == RECORDING:
                self._close_loop()
            self._buf, self._undo = self._undo, self._buf
            self.length, self._undo_len = self._undo_len, self.length
            self._has_undo = self._undo_len > 0
            if self.pos >= self.length:
                self.pos = 0.0
            if self.state in (EMPTY, OVERDUBBING):
                self.state = PLAYING
        self.post(apply)
        return available

    # -- helpers -----------------------------------------------------------

    def _prepare_snapshot(self):
        """Copy the loop into the undo buffer ahead of an overdub.

        Runs on the control thread, which is safe because the callback only
        *reads* ``_buf`` while playing or stopped. Returns a token that the
        posted closure checks, so a snapshot that went stale (the loop was
        swapped or changed in between) is never used as the undo step.
        """
        with self._lock:
            if self.state not in (PLAYING, STOPPED) or self.length <= 0:
                return None
            buf, undo, n = self._buf, self._undo, self.length
            undo[:, :n] = buf[:, :n]
            return (buf, undo, n)

    def _begin_overdub(self, snap) -> None:
        if snap is None or snap[0] is not self._buf or \
                snap[1] is not self._undo or snap[2] != self.length:
            return  # stale snapshot: the next press takes a fresh one
        self._undo_len = self.length
        self._has_undo = True
        self.state = OVERDUBBING

    def _start_recording(self) -> None:
        self.length = 0
        self.pos = 0.0
        self.state = RECORDING

    def _close_loop(self) -> None:
        self.length = int(self.pos)
        self.pos = 0.0
        if self.length > _XFADE * 2:
            # Blend the tail into the head so the seam is continuous.
            # 192 samples: cheap enough to do inside the callback.
            fade = np.linspace(0.0, 1.0, _XFADE, dtype=np.float32)
            head = self._buf[:, :_XFADE]
            tail = self._buf[:, self.length - _XFADE:self.length]
            head *= fade
            head += tail * (1.0 - fade)
            self.length -= _XFADE
        elif self.length < 64:
            self.length = 0
            self.state = EMPTY

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
                if self.length:
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
