"""Audio engine.

Owns the stream and the callback. Everything the UI wants to change either is
a plain float attribute (safe: a CPython attribute store is atomic and the
worst case is a value landing one block late) or goes through ``post()``,
a deque drained at the top of each block.

Rules the callback obeys:

*   No allocation that scales with anything but the block size.
*   No locks, no file I/O, no logging.
*   Never raises: a failure degrades to silence-plus-a-counter, so a bug shows
    up as a number in the status bar instead of a crashed stream.
"""

from __future__ import annotations

import gc
import math
import os
import threading
import time
from collections import deque

import numpy as np

# sounddevice ships two PortAudio builds and picks the non-ASIO one unless
# this is set *before* import. On this machine ASIO measured 11.4 ms round
# trip against 45 ms for WASAPI, so it is worth the environment poke.
os.environ.setdefault("SD_ENABLE_ASIO", "1")
import sounddevice as sd  # noqa: E402

from .dsp.chain import Chain
from .looper import Looper
from .recorder import Recorder
from .tuner import InputRing

BLOCK_SIZES = (64, 96, 128, 192, 256, 384, 512)
SAMPLE_RATES = (44100, 48000, 96000)


class Metronome:
    """Click track. Useful enough while looping to be worth the 30 lines."""

    def __init__(self):
        self.enabled = False
        self.bpm = 100.0
        self.beats_per_bar = 4
        self.level = 0.35
        self.sr = 48000.0
        self._click = np.zeros(0)
        self._accent = np.zeros(0)
        self._counter = 0.0
        self._beat = 0
        self._play_pos = -1
        self._play_buf = None

    def prepare(self, sr: float) -> None:
        self.sr = float(sr)
        self._click = self._make(1600.0, 0.028)
        self._accent = self._make(2400.0, 0.034)
        self._counter = 0.0
        self._beat = 0
        self._play_pos = -1

    def _make(self, freq: float, dur: float) -> np.ndarray:
        n = int(self.sr * dur)
        t = np.arange(n) / self.sr
        env = np.exp(-t * 55.0)
        return (np.sin(2 * math.pi * freq * t) * env).astype(np.float64)

    def render(self, out: np.ndarray, n: int) -> None:
        if not self.enabled:
            self._play_pos = -1
            return
        period = self.sr * 60.0 / max(self.bpm, 20.0)
        for i in range(n):
            if self._counter <= 0.0:
                self._counter += period
                self._play_buf = self._accent if self._beat == 0 else self._click
                self._play_pos = 0
                self._beat = (self._beat + 1) % max(self.beats_per_bar, 1)
            self._counter -= 1.0
            if self._play_pos >= 0 and self._play_buf is not None:
                if self._play_pos < len(self._play_buf):
                    v = self._play_buf[self._play_pos] * self.level
                    out[0, i] += v
                    out[1, i] += v
                    self._play_pos += 1
                else:
                    self._play_pos = -1


class BroadcastOut:
    """A second, independent output stream for Discord / OBS / streaming.

    ASIO talks to the hardware directly and never enters the Windows audio
    graph, so nothing that captures application audio can hear it. This opens a
    normal WASAPI stream in parallel and feeds it a copy of the mix, which
    makes Tanin audible to anything that records apps.

    It runs on its **own clock**, so it cannot share buffers with the main
    callback -- the two devices drift apart by tens of parts per million and
    would collide. A ring buffer absorbs that, plus a slow one-sample-at-a-time
    correction to stop the fill level walking into an underrun over an hour.
    Deliberately higher latency than the monitor path: this is for listeners,
    who cannot tell, not for the player, who can.
    """

    def __init__(self):
        self.enabled = False
        self.device: int | None = None
        self.level = 1.0
        self.running = False
        self.underruns = 0
        self.overruns = 0
        self.last_error = ""

        self.sr = 48000.0
        self._cap = 0
        self._ring = np.zeros((2, 0), dtype=np.float32)
        self._w = 0
        self._r = 0
        self._target = 0
        self._warmup = 0
        self._tone_left = 0
        self._tone_phase = 0.0
        self._stream: sd.OutputStream | None = None

    def prepare(self, sr: float) -> None:
        self.sr = float(sr)
        self._cap = int(sr * 0.5)
        self._ring = np.zeros((2, self._cap), dtype=np.float32)
        self._target = int(sr * 0.12)
        self._w = self._target
        self._r = 0
        self.underruns = 0
        self.overruns = 0
        # The device's first few callbacks land before the main engine has
        # pushed anything real. Counting those as gaps would put a permanent
        # warning in the status bar for a condition that always resolves.
        self._warmup = int(self.sr * 0.5)

    def start(self, device: int | None, sr: float) -> str:
        self.stop()
        if device is None:
            return "no device selected"
        self.prepare(sr)
        self.device = device
        try:
            self._stream = sd.OutputStream(
                samplerate=int(sr), blocksize=0, device=device, channels=2,
                dtype="float32", latency="high", callback=self._cb)
            self._stream.start()
            self.running = True
            self.last_error = ""
            return ""
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._stream = None
            self.running = False
            return self.last_error

    def stop(self) -> None:
        s = self._stream
        self._stream = None
        self.running = False
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass

    # -- called from the main audio callback -------------------------------

    def push(self, wet: np.ndarray, n: int) -> None:
        if not self.running or self._cap == 0:
            return
        w = self._w
        end = w + n
        g = self.level
        if end <= self._cap:
            self._ring[0, w:end] = wet[0] * g
            self._ring[1, w:end] = wet[1] * g
        else:
            first = self._cap - w
            self._ring[0, w:] = wet[0, :first] * g
            self._ring[1, w:] = wet[1, :first] * g
            rest = end - self._cap
            self._ring[0, :rest] = wet[0, first:] * g
            self._ring[1, :rest] = wet[1, first:] * g
        self._w = end % self._cap

    # -- the broadcast device's own callback -------------------------------

    def test_tone(self, seconds: float = 2.0) -> None:
        """Play a tone out of the Stream Out device only.

        Lets the setup be checked before relying on it: you should NOT hear
        this, and whoever is listening on Discord should.
        """
        self._tone_left = int(self.sr * seconds)

    def _cb(self, outdata, frames, time_info, status) -> None:
        try:
            if self._tone_left > 0:
                n = min(frames, self._tone_left)
                t = (self._tone_phase + np.arange(frames)) / self.sr
                self._tone_phase += frames
                tone = (np.sin(2 * np.pi * 660.0 * t) * 0.25).astype(np.float32)
                tone[n:] = 0.0
                outdata[:, 0] = tone
                outdata[:, 1] = tone
                self._tone_left -= n
                return
            cap = self._cap
            r = self._r
            avail = (self._w - r) % cap
            warm = self._warmup > 0
            if warm:
                self._warmup -= frames

            if avail < frames:
                # Producer has not caught up: emit silence rather than replay
                # stale audio, and let the counter say so.
                outdata[:] = 0.0
                if not warm:
                    self.underruns += 1
                self._r = (self._w - self._target) % cap
                return
            if avail > cap - frames * 2:
                # Fell too far behind; jump forward instead of wrapping past
                # the writer, which would play the buffer backwards in time.
                r = (self._w - self._target) % cap
                if not warm:
                    self.overruns += 1

            # Gentle drift correction: at most one sample per block, and only
            # outside a wide deadband. Two independent clocks differ by tens of
            # ppm, so this fires rarely and is inaudible when it does.
            skew = avail - self._target
            adjust = 0
            if skew > self.sr * 0.03:
                adjust = 1
            elif skew < -self.sr * 0.03:
                adjust = -1

            take = frames + adjust
            end = r + take
            if end <= cap:
                chunk = self._ring[:, r:end]
            else:
                chunk = np.concatenate(
                    (self._ring[:, r:], self._ring[:, : end - cap]), axis=1)
            if adjust == 1:
                chunk = chunk[:, :frames]           # drop one sample
            elif adjust == -1:
                chunk = np.concatenate((chunk, chunk[:, -1:]), axis=1)
            outdata[:, 0] = chunk[0]
            outdata[:, 1] = chunk[1]
            self._r = end % cap
        except Exception:
            outdata[:] = 0.0
            self.underruns += 1


class Engine:
    def __init__(self, recordings_dir: str):
        self.chain = Chain()
        self.looper = Looper()
        self.recorder = Recorder(recordings_dir)
        self.metronome = Metronome()
        self.broadcast = BroadcastOut()
        self.broadcast_device: int | None = None
        self.input_ring = InputRing(16384)

        self.samplerate = 48000
        self.blocksize = 256          # what the user asked for
        self.active_blocksize = 256   # what the driver actually gave us
        self.input_device = None
        self.output_device = None
        self.input_channel = 0        # which hardware input the guitar is in
        self.exclusive = True         # WASAPI exclusive mode -> real latency
        self.using_asio = False
        self.monitor_dry = False
        self.rate_fallback: tuple[int, int] | None = None
        self.exclusive_fallback = False
        self.channel_peaks = np.zeros(16, dtype=np.float64)
        self.n_input_channels = 1

        self.stream: sd.Stream | None = None
        self.running = False
        self.last_error = ""
        self.xruns = 0
        self.callback_errors = 0
        self.last_callback_exc = ""
        self.load = 0.0               # 0..1 fraction of the block deadline used
        self.peak_load = 0.0
        self.blocks_processed = 0
        self.stream_latency_ms = 0.0

        self._commands: deque = deque()
        self._lock = threading.Lock()
        self._mono_in = np.zeros(0, dtype=np.float64)
        self._deadline = 1.0

    # -- control-thread API ------------------------------------------------

    def post(self, fn) -> None:
        """Queue a callable to run at the top of the next block.

        For anything that would resize or swap a buffer. deque.append and
        popleft are atomic in CPython, so this needs no lock.

        With no stream running there is no audio thread to race and no block
        to wait for, so it runs right away. Queuing it instead would leave it
        sitting there until the next start, which used to throw it away.
        """
        if self.running and self.stream is not None:
            self._commands.append(fn)
        else:
            fn()

    def _drain_commands(self) -> None:
        """Run everything still queued. Control thread, stream not running."""
        while True:
            try:
                fn = self._commands.popleft()
            except IndexError:
                return
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                self.last_callback_exc = repr(exc)

    def _read_ir(self, path: str):
        """Read, resample and FFT an IR file. Control thread only."""
        import soundfile as sf
        data, sr = sf.read(path, dtype="float64", always_2d=True)
        ir = data[:, 0]
        if abs(sr - self.samplerate) > 1:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(sr), int(self.samplerate))
            ir = resample_poly(ir, int(self.samplerate) // g, int(sr) // g)
        return self.chain["ir"].build_kernel(ir)

    def load_ir(self, path: str) -> str:
        """Read the IR and do all its maths here, on the control thread; the
        audio thread only swaps a reference."""
        try:
            kernel = self._read_ir(path)
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
        self.post(lambda: self.chain["ir"].set_kernel(kernel, path))
        return ""

    def clear_ir(self) -> None:
        self.post(lambda: self.chain["ir"].set_kernel(None, ""))

    def apply_preset(self, state: dict) -> str:
        """Load a whole preset, including its IR file. Returns '' or an error.

        The IR file is read here (file I/O is banned on the audio thread), then
        the IR swap and every knob land together in one queued command, so the
        new sound starts on a block boundary instead of in pieces. If the IR
        file can't be read, the rest of the preset still loads, the IR block
        is switched off, and the error comes back for the UI to show.
        """
        ir_state = state.get("ir") if isinstance(state, dict) else None
        path = ir_state.get("ir_path", "") if isinstance(ir_state, dict) else ""
        path = path if isinstance(path, str) else ""
        blk = self.chain["ir"]
        err = ""
        kernel = None
        keep = bool(path) and path == blk.ir_path and blk.has_ir
        if path and not keep:
            if not os.path.isfile(path):
                err = f"IR file not found: {path}"
            else:
                try:
                    kernel = self._read_ir(path)
                except Exception as exc:  # noqa: BLE001
                    err = f"Could not load IR {os.path.basename(path)}: " \
                          f"{type(exc).__name__}: {exc}"

        def swap():
            if not keep:
                blk.set_kernel(kernel, path if kernel is not None else "")
            self.chain.apply(state)
            if not blk.has_ir:
                blk.enabled = False

        self.post(swap)
        return err

    # -- stream lifecycle --------------------------------------------------

    def _extra_settings(self):
        if not self.exclusive or self.using_asio:
            return None, None
        try:
            hostapi_in = sd.query_devices(self.input_device)["hostapi"]
            hostapi_out = sd.query_devices(self.output_device)["hostapi"]
            wasapi = next(i for i, h in enumerate(sd.query_hostapis())
                          if h["name"] == "Windows WASAPI")
        except Exception:
            return None, None
        si = sd.WasapiSettings(exclusive=True) if hostapi_in == wasapi else None
        so = sd.WasapiSettings(exclusive=True) if hostapi_out == wasapi else None
        return si, so

    def _probe_asio_blocksize(self) -> int:
        """Ask the ASIO driver what buffer size it actually intends to use.

        ASIO buffer size belongs to the driver's own control panel, not to us.
        Guessing and letting PortAudio adapt would add a buffer of latency
        silently, so we open a throwaway stream, watch one callback, and then
        size every DSP buffer to exactly that.
        """
        seen: list[int] = []

        def probe_cb(indata, outdata, frames, t, status):
            seen.append(frames)
            outdata[:] = 0.0

        try:
            s = sd.Stream(samplerate=self.samplerate, blocksize=0,
                          device=(self.input_device, self.input_device),
                          channels=(1, 2), dtype="float32", callback=probe_cb)
            s.start()
            for _ in range(40):
                if seen:
                    break
                time.sleep(0.005)
            s.stop()
            s.close()
        except Exception:
            return 0
        return max(seen) if seen else 0

    def start(self) -> str:
        """Open the stream. Returns an error string, or '' on success.

        A rejected sample rate is retried once at whatever the device says it
        wants. Windows shared-mode devices change their format behind your back
        (any other app, or the vendor's control panel, can move it), so the
        request that worked yesterday can fail today for a reason that has
        nothing to do with the user.
        """
        self.rate_fallback = None
        self.exclusive_fallback = False

        err = self._try_start()
        if not err:
            return ""

        # Retry 1: the device refused the sample rate.
        if "sample rate" in err.lower():
            try:
                native = int(sd.query_devices(self.input_device)["default_samplerate"])
            except Exception:
                native = 0
            if native and native != self.samplerate:
                requested = self.samplerate
                self.samplerate = native
                if not self._try_start():
                    self.last_error = ""
                    self.rate_fallback = (requested, native)
                    return ""
                self.samplerate = requested

        # Retry 2: exclusive mode is not always negotiable, and a working
        # shared-mode stream beats a correct-in-principle error dialog.
        if self.exclusive and not self.using_asio:
            self.exclusive = False
            if not self._try_start():
                self.last_error = ""
                self.exclusive_fallback = True
                return ""
            self.exclusive = True

        self.last_error = err
        return err

    def _try_start(self) -> str:
        self.stop()
        self.last_error = ""
        try:
            self.using_asio = is_asio(self.input_device)
            if self.using_asio:
                # ASIO is a single-device duplex API: one driver owns both
                # directions. Pairing it with some other output silently fails
                # or runs two clocks against each other.
                self.output_device = self.input_device
                probed = self._probe_asio_blocksize()
                # The ASIO driver's own control panel owns this number. We
                # report it rather than fight it, because PortAudio would
                # otherwise adapt silently and add a buffer of latency.
                self.active_blocksize = probed or self.blocksize
            else:
                self.active_blocksize = self.blocksize
            self._prepare()
            si, so = self._extra_settings()
            self.stream = sd.Stream(
                samplerate=self.samplerate,
                blocksize=self.active_blocksize,
                device=(self.input_device, self.output_device),
                channels=(self._in_channels, 2),
                dtype="float32",
                latency="low",
                callback=self._callback,
                extra_settings=(si, so) if (si or so) else None,
                finished_callback=self._on_finished,
            )
            self.stream.start()
            self.running = True
            lat = self.stream.latency
            self.stream_latency_ms = (lat[0] + lat[1]) * 1000.0
            # Refcounting alone keeps the callback clean; a generational GC
            # pass in the middle of a block is a dropout waiting to happen.
            if self.broadcast.enabled:
                self.broadcast.start(self.broadcast_device, self.samplerate)
            gc.collect()
            gc.freeze()
            gc.disable()
            return ""
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.running = False
            self.stream = None
            gc.enable()
            return self.last_error

    def stop(self) -> None:
        self.broadcast.stop()
        s = self.stream
        self.stream = None
        self.running = False
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        # Anything queued for a block that never came (a preset clicked just
        # as the stream went down) still has to land.
        self._drain_commands()
        if not gc.isenabled():
            gc.enable()
            gc.unfreeze()

    def _on_finished(self) -> None:
        self.running = False

    def _prepare(self) -> None:
        dev = sd.query_devices(self.input_device)
        self._in_channels = max(1, min(int(dev["max_input_channels"]), 8))
        if self.input_channel >= self._in_channels:
            self.input_channel = 0
        sr = float(self.samplerate)
        n = int(self.active_blocksize)
        # Was a clear() further down, which silently dropped preset and IR
        # changes made while stopped. Run them instead (no stream yet, so no
        # race), and before prepare() so it sizes whatever they swapped in.
        self._drain_commands()
        self.chain.prepare(sr, n)
        self.looper.prepare(sr, n)
        self.recorder.prepare(sr)
        self.metronome.prepare(sr)
        self._mono_in = np.zeros(n, dtype=np.float64)
        self._silence = np.zeros(n, dtype=np.float64)
        self._deadline = n / sr
        self.xruns = 0
        self.callback_errors = 0
        self.blocks_processed = 0
        self.peak_load = 0.0
        self.channel_peaks[...] = 0.0
        self.n_input_channels = self._in_channels

    # -- the callback ------------------------------------------------------

    def _callback(self, indata, outdata, frames, time_info, status) -> None:
        t0 = time.perf_counter()
        try:
            if status:
                if status.input_underflow or status.output_underflow or \
                        status.input_overflow or status.output_overflow:
                    self.xruns += 1

            # Drain queued structural changes. Bounded so a UI that spams
            # commands cannot stretch one block indefinitely.
            for _ in range(8):
                try:
                    self._commands.popleft()()
                except IndexError:
                    break
                except Exception as exc:  # noqa: BLE001
                    self.callback_errors += 1
                    self.last_callback_exc = repr(exc)

            n = frames
            if n > self.active_blocksize or n <= 0:
                # Larger than we sized for: pass silence rather than allocate.
                outdata[:] = 0.0
                return

            nch = indata.shape[1]
            # Meter every hardware input, not just the selected one. Picking
            # the right jack is guesswork otherwise -- the channel that moves
            # when you play is the answer.
            self.channel_peaks[:nch] = np.max(np.abs(indata), axis=0)
            self.n_input_channels = nch

            ch = self.input_channel
            if ch >= nch:
                ch = 0
            mono = indata[:, ch].astype(np.float64)
            self.input_ring.push(indata[:, ch], n)

            wet = self.chain.process(mono)

            if self.monitor_dry:
                wet[0] += mono * 0.5
                wet[1] += mono * 0.5

            wet = self.looper.process(wet, n)
            self.metronome.render(wet, n)
            self.recorder.push(wet, mono, n)
            self.broadcast.push(wet, n)

            np.nan_to_num(wet, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(wet, -1.0, 1.0, out=wet)
            outdata[:, 0] = wet[0]
            outdata[:, 1] = wet[1]
            self.blocks_processed += 1

        except Exception as exc:  # noqa: BLE001
            # Nothing here may propagate: PortAudio would abort the stream and
            # the user would just hear it stop. Count it, keep going.
            self.callback_errors += 1
            self.last_callback_exc = repr(exc)
            outdata[:] = 0.0
        finally:
            dt = time.perf_counter() - t0
            self.load = dt / self._deadline
            if self.load > self.peak_load:
                self.peak_load = self.load

    # -- helpers for the UI ------------------------------------------------

    def reset_meters(self) -> None:
        self.peak_load = 0.0
        self.xruns = 0

    def panic(self) -> None:
        """Kill anything ringing: delay tails, reverb, looper playback."""
        self.post(self.chain.reset)


# --------------------------------------------------------------------------
# Device discovery
# --------------------------------------------------------------------------

def hostapi_name(device: int | None) -> str:
    try:
        return sd.query_hostapis(sd.query_devices(device)["hostapi"])["name"]
    except Exception:
        return "?"


def is_asio(device: int | None) -> bool:
    return hostapi_name(device) == "ASIO"


def _paren_token(name: str) -> str:
    """The distinctive bit of a Windows device name.

    Windows names virtual cables as 'Line Out (Foo VAD Wave)' on the playback
    side and 'Microphone (Foo VAD Wave)' on the capture side -- the part in
    brackets is the actual device, and the prefix is just its role.
    """
    if "(" in name and ")" in name:
        return name[name.rfind("(") + 1: name.rfind(")")].strip()
    return name.strip()


def matching_input(out_name: str) -> str:
    """Given a playback device, find the capture device that loops back to it.

    This is what turns 'Stream Out' into usable instructions: send the tone to
    a virtual cable's playback end, then tell the user which microphone to pick
    in Discord.
    """
    token = _paren_token(out_name)
    if not token:
        return ""
    for d in list_devices("input"):
        if _paren_token(d["name"]) == token:
            return d["name"]
    return ""


def is_virtual_cable(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in (
        "cable", "voicemeeter", "vb-audio", "vad ", "_vad", "virtual"))


# Words that say what a device *does*, not which box it is. Comparing device
# names without stripping these makes every pair of speakers look identical.
_GENERIC_WORDS = {
    "speakers", "speaker", "line", "out", "output", "input", "microphone",
    "mic", "headphones", "headphone", "asio", "driver", "wave", "usb",
    "audio", "device", "default", "primary", "sound", "digital", "analog",
}


def _hardware_words(name: str) -> set[str]:
    import re
    return {w for w in re.findall(r"[a-z0-9]+", name.lower())
            if len(w) >= 3 and w not in _GENERIC_WORDS}


def same_hardware(a: str, b: str) -> bool:
    """Whether two device names refer to the same physical box.

    'AudioBox ASIO Driver' and 'Speakers (AudioBox USB 96)' are the same
    interface under two host APIs. Streaming to one while ASIO holds the other
    either fails or fights over the device, so this is worth detecting.
    """
    return bool(_hardware_words(a) & _hardware_words(b))


def broadcast_candidates(monitor_device: int | None) -> list[dict]:
    """Output devices worth offering for Stream Out, best first.

    Devices that are the same hardware as the monitor path are excluded
    outright. Sending a second copy to the interface you are already
    monitoring through gives you the same guitar twice, ~120 ms apart -- an
    echo, not a feature.

    Ordering prefers *inaudible* endpoints. Whatever is picked here plays out
    of that device for real, so a physical output means hearing the guitar a
    second time, late. A virtual sink nobody listens to gives Discord an audio
    session to capture and costs the player nothing.
    """
    try:
        monitor_name = sd.query_devices(monitor_device)["name"]
    except Exception:
        monitor_name = ""

    out = []
    seen = set()
    for d in list_devices("output"):
        if d["api"] not in ("Windows WASAPI", "Windows WDM-KS"):
            continue
        if d["channels"] < 2:
            continue
        if monitor_name and same_hardware(d["name"], monitor_name):
            continue
        key = (d["name"], d["api"])
        if key in seen:
            continue
        seen.add(key)
        cable_in = matching_input(d["name"])
        d = dict(d)
        d["cable_input"] = cable_in
        d["is_cable"] = is_virtual_cable(d["name"]) and bool(cable_in)
        out.append(d)

    out.sort(key=lambda d: (_sink_score(d["name"]), d["rank"], d["name"]))
    return out


def _sink_score(name: str) -> int:
    """Lower is safer to send audio into without hearing it.

    'NVIDIA Broadcast' is deliberately *not* treated as silent: it is a
    processing endpoint that forwards to a real output, so it usually is
    audible despite looking virtual.
    """
    n = name.lower()
    if "steam streaming" in n:
        return 0
    if any(k in n for k in ("cable", "voicemeeter", "vb-audio", "vdvad",
                            "vad ", "_vad", "virtual")):
        return 1
    return 2


def suggest_broadcast_device(monitor_device: int | None) -> int | None:
    cands = broadcast_candidates(monitor_device)
    return cands[0]["index"] if cands else None


def find_device(name: str, api: str, kind: str) -> int | None:
    """Resolve a saved device by name + host API rather than by index.

    PortAudio device indices are not stable: they shift when a host API is
    added or removed (enabling ASIO renumbers everything), and when a USB
    interface is plugged in or out. A stored index that silently points at a
    different device is exactly the kind of failure nobody debugs -- it just
    looks like "the app stopped hearing my guitar".
    """
    if not name:
        return None
    devices = list_devices(kind)
    for d in devices:
        if d["name"] == name and d["api"] == api:
            return d["index"]
    for d in devices:            # same box, host API no longer available
        if d["name"] == name:
            return d["index"]
    return None


def list_devices(kind: str) -> list[dict]:
    """kind is 'input' or 'output'. Returns the useful subset, best APIs first."""
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    apis = {i: h["name"] for i, h in enumerate(sd.query_hostapis())}
    # WASAPI first (exclusive mode is the low-latency path on Windows without
    # a vendor ASIO driver), then ASIO if this PortAudio build has it.
    rank = {"ASIO": 0, "Windows WASAPI": 1, "Windows WDM-KS": 2,
            "Windows DirectSound": 3, "MME": 4}
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d[key] < 1:
            continue
        api = apis.get(d["hostapi"], "?")
        out.append({
            "index": i,
            "name": d["name"],
            "api": api,
            "channels": d[key],
            "default_sr": int(d["default_samplerate"]),
            "rank": rank.get(api, 9),
        })
    out.sort(key=lambda d: (_sink_score(d["name"]), d["rank"], d["name"]))
    return out


def _sink_score(name: str) -> int:
    """Lower is safer to send audio into without hearing it.

    'NVIDIA Broadcast' is deliberately *not* treated as silent: it is a
    processing endpoint that forwards to a real output, so it usually is
    audible despite looking virtual.
    """
    n = name.lower()
    if "steam streaming" in n:
        return 0
    if any(k in n for k in ("cable", "voicemeeter", "vb-audio", "vdvad",
                            "vad ", "_vad", "virtual")):
        return 1
    return 2


def default_devices() -> tuple[int | None, int | None]:
    """Prefer a real interface on WASAPI over whatever Windows calls default."""
    ins = list_devices("input")
    outs = list_devices("output")
    pick_in = ins[0]["index"] if ins else None
    pick_out = outs[0]["index"] if outs else None
    # If the same hardware appears on both sides, keep them paired.
    for i in ins:
        base = i["name"].split("(")[-1]
        for o in outs:
            if base and base in o["name"] and i["rank"] <= 1 and o["rank"] <= 1:
                return i["index"], o["index"]
    return pick_in, pick_out
