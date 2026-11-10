"""The effect blocks.

Convention:

*   Mono blocks take and return ``(n,)`` float64.
*   Stereo blocks take and return ``(2, n)`` float64.
*   ``prepare(sr, max_block)`` allocates. ``process()`` does not allocate
    anything that scales with anything but the block.
*   Every block has ``enabled``; when false it must be *transparent*, not
    merely quiet. The offline harness asserts bit-exactness on bypass.

Parameters are plain attributes. The UI writes them from another thread and
the audio thread reads them once per block; in CPython a float attribute store
is atomic, so no lock is needed and none is used. Anything that would resize
a buffer goes through a command queue instead (see engine.py).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from .core import (
    Biquad,
    BiquadCascade,
    DCBlocker,
    DelayLine,
    LFO,
    OnePole,
    Oversampler,
    SHAPERS,
    SmoothedParam,
    clampf,
    db2lin,
)


class Block:
    """Base class. Subclasses override prepare/reset/_process."""

    name = "block"
    stereo = False

    def __init__(self):
        self.enabled = True
        self.sr = 48000.0
        self.max_block = 512

    def prepare(self, sr: float, max_block: int) -> None:
        self.sr = float(sr)
        self.max_block = int(max_block)

    def reset(self) -> None:
        pass

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x
        n = x.shape[-1]
        if n > self.max_block or n == 0:
            # Larger than we sized for: hand it on untouched rather than
            # allocate on the audio thread. One un-effected block beats a click.
            return x
        return self._process(x, n)

    def _process(self, x: np.ndarray, n: int) -> np.ndarray:  # pragma: no cover
        return x


# ==========================================================================
# Noise gate
# ==========================================================================

class NoiseGate(Block):
    name = "gate"

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.threshold_db = -55.0
        self.attack_ms = 2.0
        self.release_ms = 120.0
        self._det = OnePole(3.0)
        self._smooth = OnePole(10.0)
        self._rel = OnePole(120.0)

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        for f in (self._det, self._smooth, self._rel):
            f.prepare(sr, 1)
            f.reset()

    def reset(self):
        for f in (self._det, self._smooth, self._rel):
            f.reset()

    def _process(self, x, n):
        self._det.set_tau(1.0)
        env = self._det.process(np.abs(x))
        thr = db2lin(self.threshold_db)
        # Soft knee over 6 dB so a decaying note fades rather than chops.
        lo = thr * 0.5
        g = np.clip((env - lo) / (thr - lo + 1e-12), 0.0, 1.0)
        g *= g  # square the curve -> gentler open, faster close

        # Asymmetric ballistics without a per-sample branch: run a fast and a
        # slow smoother and take the larger. A rising target is tracked by the
        # fast one (snappy open); a falling target is held up by the slow one
        # (long, natural close). Choosing per block instead would make the
        # result depend on buffer size, which is a bug you only hear after
        # someone changes their latency setting.
        self._smooth.set_tau(self.attack_ms)
        self._rel.set_tau(self.release_ms)
        gs = np.maximum(self._smooth.process(g), self._rel.process(g))
        return x * gs


# ==========================================================================
# Compressor
# ==========================================================================

class Compressor(Block):
    name = "comp"

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.threshold_db = -20.0
        self.ratio = 4.0
        self.attack_ms = 8.0
        self.release_ms = 150.0
        self.makeup_db = 0.0
        self.mix = 1.0
        self.gain_reduction_db = 0.0  # telemetry for the UI meter
        self._det = OnePole(1.0)
        self._att = OnePole(8.0)
        self._rel = OnePole(150.0)
        self._gain = SmoothedParam(1.0)

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        for f in (self._det, self._att, self._rel):
            f.prepare(sr, 1)
        self._gain.prepare(max_block)
        self.reset()

    def reset(self):
        self._det.reset()
        self._att.reset()
        self._rel.reset(0.0)

    def _process(self, x, n):
        env = self._det.process(np.abs(x))
        env_db = 20.0 * np.log10(env + 1e-9)
        over = env_db - self.threshold_db
        # Soft knee, 6 dB wide.
        knee = 6.0
        gr_db = np.where(
            over <= -knee / 2, 0.0,
            np.where(
                over >= knee / 2,
                over * (1.0 / self.ratio - 1.0),
                ((1.0 / self.ratio - 1.0) * (over + knee / 2) ** 2) / (2 * knee),
            ),
        )
        # Attack/release ballistics on the gain-reduction curve.
        self._att.set_tau(self.attack_ms)
        self._rel.set_tau(self.release_ms)
        fast = self._att.process(gr_db)
        slow = self._rel.process(gr_db)
        # Take whichever is currently *more* reduction: attack wins going down,
        # release wins coming back up.
        smoothed = np.minimum(fast, slow)
        self.gain_reduction_db = float(smoothed[-1])
        g = 10.0 ** ((smoothed + self.makeup_db) / 20.0)
        wet = x * g
        if self.mix >= 0.999:
            return wet
        return x * (1.0 - self.mix) + wet * self.mix


# ==========================================================================
# Drive pedal
# ==========================================================================

DRIVE_MODELS = {
    # name        shaper   hp_hz  lp_hz  gain_range  bias   post_lp
    "Screamer": ("diode", 720.0, 6000.0, (1.0, 90.0), 0.00, 5500.0),
    "Boost":    ("tanh",  120.0, 9000.0, (1.0, 12.0), 0.00, 9000.0),
    "Rat":      ("diode", 240.0, 7000.0, (2.0, 400.0), 0.05, 4500.0),
    "Muff":     ("hard",  180.0, 6000.0, (4.0, 600.0), 0.02, 3500.0),
    "Fuzz":     ("fold",   90.0, 8000.0, (6.0, 900.0), 0.18, 4000.0),
}


class Drive(Block):
    name = "drive"

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.model = "Screamer"
        self.drive = 0.35       # 0..1
        self.tone = 0.5         # 0..1
        self.level = 0.5        # 0..1
        self.mix = 1.0
        self._hp = Biquad("highpass", 720.0, 0.707)
        self._tone_lp = Biquad("lowpass", 5500.0, 0.707)
        self._post_lp = Biquad("lowpass", 6000.0, 0.707)
        self._dc = DCBlocker()
        self._os = Oversampler(factor=4, order=6)
        self._out = SmoothedParam(1.0)

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._hp.prepare(sr, 1)
        self._tone_lp.prepare(sr, 1)
        self._post_lp.prepare(sr, 1)
        self._dc.prepare(sr, 1)
        self._os.prepare(sr, max_block)
        self._out.prepare(max_block)
        self._out.snap(1.0)

    def reset(self):
        self._hp.reset()
        self._tone_lp.reset()
        self._post_lp.reset()
        self._dc.reset()
        self._os.reset()

    def _process(self, x, n):
        shaper_name, hp, lp, grange, bias, post_lp = DRIVE_MODELS.get(
            self.model, DRIVE_MODELS["Screamer"]
        )
        shaper = SHAPERS[shaper_name]
        self._hp.set(f0=hp)
        pre = self._hp.process(x)

        gmin, gmax = grange
        gain = gmin * (gmax / gmin) ** clampf(self.drive, 0.0, 1.0)

        up = self._os.up(pre)
        if shaper_name == "tube":
            shaped = shaper(up, gain, bias)
        else:
            shaped = shaper(up, gain)
            if bias:
                shaped = shaped + bias * 0.5
        wet = self._os.down(shaped)
        wet = self._dc.process(wet)

        # Tone: sweep the post lowpass rather than a fixed tilt.
        self._tone_lp.set(f0=400.0 * (12.0 ** clampf(self.tone, 0.0, 1.0)))
        self._post_lp.set(f0=post_lp)
        wet = self._post_lp.process(self._tone_lp.process(wet))

        # Compensate for the huge gain swing so the level knob stays useful.
        comp = 1.0 / (1.0 + math.log10(gain) * 0.35)
        self._out.set(db2lin(-20.0 + 26.0 * clampf(self.level, 0.0, 1.0)) * comp * 4.0)
        wet = wet * self._out.ramp(n)

        if self.mix >= 0.999:
            return wet
        return x * (1.0 - self.mix) + wet * self.mix


# ==========================================================================
# Amplifier
# ==========================================================================

class AmpModel:
    """Static voicing description for one amp. No state, safe to share."""

    def __init__(self, name, stages, stage_gain, bias, input_hp, coupling_lp,
                 tone_bass, tone_mid, tone_mid_q, tone_treble, out_lp, gain_range,
                 shaper="tube"):
        self.name = name
        self.stages = stages
        self.stage_gain = stage_gain
        self.bias = bias
        self.input_hp = input_hp
        self.coupling_lp = coupling_lp
        self.tone_bass = tone_bass
        self.tone_mid = tone_mid
        self.tone_mid_q = tone_mid_q
        self.tone_treble = tone_treble
        self.out_lp = out_lp
        self.gain_range = gain_range
        self.shaper = shaper


AMP_MODELS = {
    "Clean": AmpModel(
        "Clean", stages=2, stage_gain=0.65, bias=0.06, input_hp=45.0,
        coupling_lp=12000.0, tone_bass=95.0, tone_mid=500.0, tone_mid_q=0.6,
        tone_treble=3200.0, out_lp=9000.0, gain_range=(0.8, 14.0)),
    "Crunch": AmpModel(
        "Crunch", stages=3, stage_gain=0.8, bias=0.10, input_hp=70.0,
        coupling_lp=9000.0, tone_bass=110.0, tone_mid=650.0, tone_mid_q=0.7,
        tone_treble=2800.0, out_lp=7000.0, gain_range=(1.5, 60.0)),
    "Lead": AmpModel(
        "Lead", stages=4, stage_gain=0.95, bias=0.13, input_hp=90.0,
        coupling_lp=7000.0, tone_bass=120.0, tone_mid=750.0, tone_mid_q=0.8,
        tone_treble=2600.0, out_lp=6200.0, gain_range=(3.0, 220.0)),
    "Metal": AmpModel(
        "Metal", stages=4, stage_gain=1.05, bias=0.08, input_hp=115.0,
        coupling_lp=6000.0, tone_bass=130.0, tone_mid=800.0, tone_mid_q=1.0,
        tone_treble=3000.0, out_lp=5600.0, gain_range=(6.0, 600.0)),
}


class Amp(Block):
    """Cascaded-gain-stage preamp, passive-style tone stack, power-amp sag."""

    name = "amp"

    def __init__(self):
        super().__init__()
        self.enabled = True
        self.model = "Crunch"
        self.gain = 0.45
        self.bass = 0.5
        self.mid = 0.5
        self.treble = 0.5
        self.presence = 0.35
        self.master = 0.5
        self.sag = 0.3

        self._in_hp = Biquad("highpass", 70.0, 0.707)
        self._couple = [Biquad("lowpass", 9000.0, 0.707) for _ in range(4)]
        self._stage_hp = [Biquad("highpass", 30.0, 0.707) for _ in range(4)]
        self._bass = Biquad("lowshelf", 110.0, 0.707)
        self._mid = Biquad("peak", 650.0, 0.7)
        self._treble = Biquad("highshelf", 2800.0, 0.707)
        self._presence = Biquad("highshelf", 5200.0, 0.707)
        self._out_lp = Biquad("lowpass", 7000.0, 0.707)
        self._dc = DCBlocker()
        self._os = Oversampler(factor=4, order=6)
        self._sag_env = OnePole(60.0)
        self._master = SmoothedParam(0.5)
        self.output_level = 0.0  # telemetry

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        for f in (self._in_hp, self._bass, self._mid, self._treble,
                  self._presence, self._out_lp):
            f.prepare(sr, 1)
        for f in self._couple:
            f.prepare(sr, 1)
        for f in self._stage_hp:
            f.prepare(sr, 1)
        self._dc.prepare(sr, 1)
        self._os.prepare(sr, max_block)
        self._sag_env.prepare(sr, 1)
        self._sag_env.reset()
        self._master.prepare(max_block)
        self._master.snap(db2lin(-12.0))

    def reset(self):
        for f in (self._in_hp, self._bass, self._mid, self._treble,
                  self._presence, self._out_lp):
            f.reset()
        for f in self._couple:
            f.reset()
        for f in self._stage_hp:
            f.reset()
        self._dc.reset()
        self._os.reset()
        self._sag_env.reset()

    def _process(self, x, n):
        m = AMP_MODELS.get(self.model, AMP_MODELS["Crunch"])
        shaper = SHAPERS[m.shaper]

        self._in_hp.set(f0=m.input_hp)
        sig = self._in_hp.process(x)

        gmin, gmax = m.gain_range
        total = gmin * (gmax / gmin) ** clampf(self.gain, 0.0, 1.0)
        per_stage = total ** (1.0 / m.stages) * m.stage_gain

        # Power-amp sag: a loud passage pulls the rail down, which compresses
        # the attack and is a big part of why a cranked amp "breathes".
        if self.sag > 0.001:
            env = self._sag_env.process(np.abs(sig))
            sag_gain = 1.0 / (1.0 + env * self.sag * 3.0)
        else:
            sag_gain = 1.0

        up = self._os.up(sig)
        # Coupling filters run at base rate, so the per-stage lowpass is applied
        # after downsampling in a single pass; inside the oversampled loop we
        # only shape, which is where the aliasing would come from.
        for i in range(m.stages):
            up = shaper(up, per_stage, m.bias * (1.0 if i % 2 == 0 else -0.6))
            up *= 0.85  # inter-stage attenuation, keeps later stages in range
        sig = self._os.down(up)
        sig = self._dc.process(sig)

        self._couple[0].set(f0=m.coupling_lp)
        sig = self._couple[0].process(sig)
        sig = sig * sag_gain

        # Tone stack. +/-15 dB, centred at 0.5.
        self._bass.set(f0=m.tone_bass, gain_db=(self.bass - 0.5) * 30.0)
        self._mid.set(f0=m.tone_mid, q=m.tone_mid_q, gain_db=(self.mid - 0.5) * 30.0)
        self._treble.set(f0=m.tone_treble, gain_db=(self.treble - 0.5) * 30.0)
        self._presence.set(f0=5200.0, gain_db=clampf(self.presence, 0.0, 1.0) * 14.0)
        self._out_lp.set(f0=m.out_lp)

        sig = self._bass.process(sig)
        sig = self._mid.process(sig)
        sig = self._treble.process(sig)
        sig = self._presence.process(sig)
        sig = self._out_lp.process(sig)

        self._master.set(db2lin(-40.0 + 46.0 * clampf(self.master, 0.0, 1.0)))
        sig = sig * self._master.ramp(n)
        self.output_level = float(np.max(np.abs(sig))) if n else 0.0
        return sig


# ==========================================================================
# Cabinet
# ==========================================================================

CAB_MODELS = {
    # (hp_hz, hp_q, lp_hz, lp_q, [(peak_hz, q, db), ...])
    "1x12 Combo": (95.0, 0.8, 4800.0, 0.8,
                   [(180.0, 1.2, 2.5), (900.0, 1.4, -4.0), (2300.0, 1.6, 5.0),
                    (3800.0, 2.0, -6.0)]),
    "2x12 Vintage": (85.0, 0.9, 4200.0, 0.9,
                     [(140.0, 1.0, 3.0), (700.0, 1.3, -5.0), (2000.0, 1.5, 4.0),
                      (3400.0, 2.2, -7.0)]),
    "4x12 Modern": (78.0, 1.0, 4600.0, 1.1,
                    [(110.0, 1.0, 4.5), (450.0, 1.2, -3.0), (800.0, 1.8, -6.5),
                     (2400.0, 1.4, 5.5), (4000.0, 2.5, -8.0)]),
    "4x12 Classic": (82.0, 0.9, 4000.0, 0.9,
                     [(120.0, 1.0, 3.5), (500.0, 1.1, -2.0), (1600.0, 1.3, 3.0),
                      (2800.0, 1.8, -5.0)]),
    "1x8 Lo-Fi": (220.0, 0.9, 3200.0, 1.0,
                  [(400.0, 1.0, 2.0), (1200.0, 1.5, 4.0), (2600.0, 2.0, -6.0)]),
}

MIC_TILT = {
    "Dynamic": (0.0, 0.0),      # (low shelf dB @ 200, high shelf dB @ 4k)
    "Ribbon": (2.0, -4.0),
    "Condenser": (-1.0, 3.5),
}


class Cabinet(Block):
    """Speaker/mic voicing.

    A real cab IR is an FIR; this is a biquad cascade fitted by ear to the same
    shape. It costs a fraction of a convolution and, unlike a fixed IR, the mic
    position control is continuous. Real ``.wav`` IRs load through IRLoader.
    """

    name = "cab"

    def __init__(self):
        super().__init__()
        self.enabled = True
        self.model = "4x12 Classic"
        self.mic = "Dynamic"
        self.position = 0.5   # 0 = cone edge (dark), 1 = dust cap (bright)
        self.distance = 0.3   # adds low-mid proximity as it goes down
        self._hp = Biquad("highpass", 82.0, 0.9)
        self._lp = Biquad("lowpass", 4000.0, 0.9)
        self._peaks = [Biquad("peak", 1000.0, 1.0, 0.0) for _ in range(5)]
        self._mic_lo = Biquad("lowshelf", 200.0, 0.707, 0.0)
        self._mic_hi = Biquad("highshelf", 4000.0, 0.707, 0.0)
        self._prox = Biquad("lowshelf", 150.0, 0.707, 0.0)
        self._tilt = Biquad("highshelf", 2500.0, 0.707, 0.0)
        self._n_active = 4

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        for f in [self._hp, self._lp, self._mic_lo, self._mic_hi,
                  self._prox, self._tilt] + self._peaks:
            f.prepare(sr, 1)

    def reset(self):
        for f in [self._hp, self._lp, self._mic_lo, self._mic_hi,
                  self._prox, self._tilt] + self._peaks:
            f.reset()

    def _process(self, x, n):
        hp, hpq, lp, lpq, peaks = CAB_MODELS.get(self.model, CAB_MODELS["4x12 Classic"])
        self._hp.set(f0=hp, q=hpq)
        self._lp.set(f0=lp, q=lpq)
        sig = self._lp.process(self._hp.process(x))

        self._n_active = len(peaks)
        for i, (f0, q, g) in enumerate(peaks):
            self._peaks[i].set(f0=f0, q=q, gain_db=g)
            sig = self._peaks[i].process(sig)

        lo_db, hi_db = MIC_TILT.get(self.mic, (0.0, 0.0))
        self._mic_lo.set(f0=200.0, gain_db=lo_db)
        self._mic_hi.set(f0=4000.0, gain_db=hi_db)
        sig = self._mic_hi.process(self._mic_lo.process(sig))

        # Mic position: off-axis at the cone edge loses top end.
        self._tilt.set(f0=2500.0, gain_db=(clampf(self.position, 0.0, 1.0) - 0.5) * 12.0)
        # Proximity effect: close mic = more low end.
        self._prox.set(f0=150.0, gain_db=(1.0 - clampf(self.distance, 0.0, 1.0)) * 6.0)
        return self._prox.process(self._tilt.process(sig))


class IRLoader(Block):
    """Overlap-save FFT convolution with a user-supplied impulse response.

    The IR is swapped in by the engine's command queue, never here, because
    changing it reallocates. Latency added is zero: overlap-save keeps the
    output aligned with the input.
    """

    name = "ir"

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.mix = 1.0
        self.gain_db = 0.0
        self.ir_path = ""
        self._h_fft = None
        self._fft_size = 0
        self._overlap = None
        self._ir_len = 0

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._rebuild()

    def set_ir(self, ir: np.ndarray | None, path: str = "") -> None:
        """Called from the engine command drain, i.e. still on the audio
        thread but before processing -- so keep it to one allocation and
        accept it. Called with None to clear."""
        self.ir_path = path
        if ir is None or ir.size == 0:
            self._h_fft = None
            self.enabled = False
            return
        ir = np.asarray(ir, dtype=np.float64).ravel()
        # 4096 taps is ~85 ms at 48k -- longer than any guitar cab IR needs.
        ir = ir[:4096]
        peak = float(np.max(np.abs(ir)))
        if peak > 0:
            ir = ir / peak * 0.5
        self._ir = ir
        self._rebuild()

    def _rebuild(self):
        ir = getattr(self, "_ir", None)
        if ir is None:
            self._h_fft = None
            return
        self._ir_len = len(ir)
        size = 1
        while size < self.max_block + self._ir_len:
            size <<= 1
        self._fft_size = size
        self._h_fft = np.fft.rfft(ir, size)
        self._hist = np.zeros(size, dtype=np.float64)

    def reset(self):
        if self._h_fft is not None:
            self._hist[...] = 0.0

    def _process(self, x, n):
        if self._h_fft is None:
            return x
        # Overlap-save: keep the previous (fft_size - n) input samples.
        self._hist[:-n] = self._hist[n:]
        self._hist[-n:] = x
        y = np.fft.irfft(np.fft.rfft(self._hist) * self._h_fft, self._fft_size)
        wet = y[-n:] * db2lin(self.gain_db)
        if self.mix >= 0.999:
            return wet
        return x * (1.0 - self.mix) + wet * self.mix


# ==========================================================================
# Modulation
# ==========================================================================

class Modulation(Block):
    name = "mod"
    stereo = True

    MODES = ("Chorus", "Flanger", "Phaser", "Tremolo", "Vibrato")

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.mode = "Chorus"
        self.rate = 0.6      # Hz-ish, mapped per mode
        self.depth = 0.5
        self.feedback = 0.3
        self.mix = 0.5
        self.stereo_spread = 0.5
        self._lfo = LFO()
        self._line = None
        self._ap_zi = None
        self._ap_stages = 6
        self._chunk = 32
        self._ph_fb = None

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._chunk = max(8, min(32, max_block))
        self._lfo.prepare(sr, max_block)
        max_delay = int(sr * 0.05) + max_block + 8
        self._line = DelayLine(max_delay, channels=2)
        self._line.prepare(max_block)
        self._ap_zi = np.zeros((2, self._ap_stages, 1), dtype=np.float64)
        self._ph_fb = np.zeros((2, self._chunk), dtype=np.float64)

    def reset(self):
        if self._line is not None:
            self._line.reset()
        if self._ap_zi is not None:
            self._ap_zi[...] = 0.0
        if self._ph_fb is not None:
            self._ph_fb[...] = 0.0
        self._lfo.phase = 0.0

    # -- individual modes ---------------------------------------------------

    def _chorus(self, x, n):
        rate = 0.05 + self.rate * 4.0
        depth_ms = 1.0 + self.depth * 7.0
        # The shortest excursion must still be at least one block behind the
        # write head (see DelayLine's ordering contract), so the centre delay
        # is pushed out at large buffer sizes rather than allowed to alias.
        min_ms = (n / self.sr) * 1000.0 + 0.5
        base_ms = max(14.0, min_ms + depth_ms)
        spread = 0.25 * clampf(self.stereo_spread, 0.0, 1.0)
        ph = self._lfo.phase_block(rate, n)
        l = np.sin(2 * math.pi * ph)
        r = np.sin(2 * math.pi * (ph + spread))
        dl = (base_ms + depth_ms * l) * 1e-3 * self.sr
        dr = (base_ms + depth_ms * r) * 1e-3 * self.sr
        wl = self._line.read_modulated(dl, n)[0]
        wr = self._line.read_modulated(dr, n)[1]
        self._line.write(x)
        return np.stack((wl, wr))

    def _flanger(self, x, n):
        """Short modulated delay with feedback.

        Feedback across a delay shorter than the block cannot be done
        block-at-a-time, so this runs in fixed sub-chunks. The chunk size sets
        the *minimum* delay, and a flanger lives at short delays -- so the
        chunk is deliberately small (32 samples, 0.67 ms at 48 k) even though
        that costs more numpy calls per block.
        """
        rate = 0.05 + self.rate * 2.0
        c = self._chunk
        fb = clampf(self.feedback, 0.0, 0.95) * 0.85
        out = np.empty((2, n), dtype=np.float64)
        base_ms = max(0.8, (c / self.sr) * 1000.0 + 0.05)
        depth_ms = self.depth * 6.0
        spread = 0.5 * clampf(self.stereo_spread, 0.0, 1.0)
        pos = 0
        while pos < n:
            k = min(c, n - pos)
            seg = x[:, pos:pos + k]
            ph = self._lfo.phase_block(rate, k)
            l = np.sin(2 * math.pi * ph)
            r = np.sin(2 * math.pi * (ph + spread))
            dl = (base_ms + depth_ms * (0.5 + 0.5 * l)) * 1e-3 * self.sr
            dr = (base_ms + depth_ms * (0.5 + 0.5 * r)) * 1e-3 * self.sr
            wl = self._line.read_modulated(dl, k)[0]
            wr = self._line.read_modulated(dr, k)[1]
            wet = np.stack((wl, wr))
            self._line.write(seg + wet * fb)
            out[:, pos:pos + k] = wet
            pos += k
        return out

    def _phaser(self, x, n):
        rate = 0.05 + self.rate * 3.0
        c = self._chunk
        stages = self._ap_stages
        fb = clampf(self.feedback, 0.0, 0.9) * 0.7
        out = np.empty((2, n), dtype=np.float64)
        spread = 0.5 * clampf(self.stereo_spread, 0.0, 1.0)
        lo = 250.0
        hi = 250.0 * (12.0 ** (0.3 + 0.7 * clampf(self.depth, 0.0, 1.0)))
        pos = 0
        while pos < n:
            k = min(c, n - pos)
            ph = self._lfo.phase_block(rate, k)
            mid_phase = float(ph[k // 2])
            for ch in range(2):
                # The coefficient is held constant across the chunk so the
                # whole stage collapses into one lfilter call. At 0.67 ms per
                # chunk an LFO below ~5 Hz cannot move far enough for the
                # stepping to be audible.
                mod = math.sin(2 * math.pi * (mid_phase + (spread if ch else 0.0)))
                fc = lo * (hi / lo) ** (0.5 + 0.5 * mod)
                t = math.tan(math.pi * clampf(fc, 20.0, self.sr * 0.45) / self.sr)
                a = (t - 1.0) / (t + 1.0)
                b = np.array([a, 1.0])
                aa = np.array([1.0, a])
                y = x[ch, pos:pos + k] + self._ph_fb[ch, :k] * fb
                for s in range(stages):
                    y, self._ap_zi[ch, s] = lfilter(b, aa, y, zi=self._ap_zi[ch, s])
                out[ch, pos:pos + k] = y
                self._ph_fb[ch, :k] = y
            pos += k
        return out

    def _tremolo(self, x, n):
        rate = 0.2 + self.rate * 11.0
        l = self._lfo.block(rate, n)
        d = clampf(self.depth, 0.0, 1.0)
        if self.stereo_spread > 0.5:
            # Auto-pan: channels move in opposition.
            gl = 1.0 - d * (0.5 + 0.5 * l)
            gr = 1.0 - d * (0.5 - 0.5 * l)
            return np.stack((x[0] * gl, x[1] * gr))
        g = 1.0 - d * (0.5 + 0.5 * l)
        return x * g

    def _process(self, x, n):
        mode = self.mode
        if mode == "Tremolo":
            wet = self._tremolo(x, n)
            return wet  # amplitude modulation is 100% wet by definition
        if mode == "Chorus":
            wet = self._chorus(x, n)
        elif mode == "Vibrato":
            return self._chorus(x, n)   # 100% wet: pitch wobble, no comb
        elif mode == "Flanger":
            wet = self._flanger(x, n)
        elif mode == "Phaser":
            wet = self._phaser(x, n)
        else:
            return x
        m = clampf(self.mix, 0.0, 1.0)
        return x * (1.0 - m) + wet * m


# ==========================================================================
# Delay
# ==========================================================================

class Delay(Block):
    name = "delay"
    stereo = True

    MODES = ("Digital", "Analog", "Tape", "Ping-Pong")

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.mode = "Analog"
        self.time_ms = 380.0
        self.feedback = 0.35
        self.mix = 0.3
        self.tone = 0.5          # damping in the repeats
        self.modulation = 0.15   # tape wow
        self._line = None
        self._damp = Biquad("lowpass", 5000.0, 0.707)
        self._damp_hp = Biquad("highpass", 120.0, 0.707)
        self._lfo = LFO()
        self._max_delay = 0

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._max_delay = int(sr * 2.2)
        self._line = DelayLine(self._max_delay + max_block + 16, channels=2)
        self._line.prepare(max_block)
        self._damp.prepare(sr, 2)
        self._damp_hp.prepare(sr, 2)
        self._lfo.prepare(sr, max_block)

    def reset(self):
        if self._line is not None:
            self._line.reset()
        self._damp.reset()
        self._damp_hp.reset()

    def _process(self, x, n):
        # A block-at-a-time feedback loop needs the delay to be at least one
        # block long. 20 ms is already below anything anyone calls a delay.
        min_ms = max(20.0, (n / self.sr) * 1000.0 + 1.0)
        t_ms = clampf(self.time_ms, min_ms, 2000.0)

        if self.modulation > 0.001 and self.mode == "Tape":
            wow = self._lfo.block(0.7, n)
            d = (t_ms * 1e-3 * self.sr) + wow * self.modulation * 0.004 * self.sr
            np.clip(d, n + 2, self._max_delay, out=d)
            wet = self._line.read_modulated(d, n)
        else:
            self._lfo.advance_only(0.7, n)
            d_int = int(t_ms * 1e-3 * self.sr)
            d_int = max(d_int, n + 2)
            wet = self._line.read_fixed(d_int, n).copy()

        # Repeat voicing.
        if self.mode == "Analog":
            self._damp.set(f0=1200.0 * (6.0 ** clampf(self.tone, 0.0, 1.0)))
            self._damp_hp.set(f0=180.0)
        elif self.mode == "Tape":
            self._damp.set(f0=900.0 * (7.0 ** clampf(self.tone, 0.0, 1.0)))
            self._damp_hp.set(f0=140.0)
        else:
            self._damp.set(f0=2000.0 * (9.0 ** clampf(self.tone, 0.0, 1.0)))
            self._damp_hp.set(f0=60.0)
        fed = self._damp_hp.process(self._damp.process(wet))
        if self.mode in ("Analog", "Tape"):
            np.tanh(fed * 1.4, out=fed)
            fed *= 0.714

        fb = clampf(self.feedback, 0.0, 0.95)
        if self.mode == "Ping-Pong":
            new = np.empty_like(fed)
            new[0] = x[0] + fed[1] * fb
            new[1] = x[1] + fed[0] * fb
        else:
            new = x + fed * fb
        self._line.write(new)

        m = clampf(self.mix, 0.0, 1.0)
        return x + wet * m


# ==========================================================================
# Reverb (Freeverb topology)
# ==========================================================================

_COMB_TUNING = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
_ALLPASS_TUNING = (556, 441, 341, 225)
_STEREO_SPREAD = 23


class Reverb(Block):
    name = "reverb"
    stereo = True

    MODES = ("Room", "Hall", "Plate", "Cathedral")

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.mode = "Hall"
        self.size = 0.6
        self.damping = 0.45
        self.mix = 0.25
        self.width = 0.8
        self.predelay_ms = 15.0
        self._comb_lines = None
        self._ap_lines = None
        self._chunk = 64

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        scale = sr / 44100.0
        self._comb_len = [[int(t * scale), int((t + _STEREO_SPREAD) * scale)]
                          for t in _COMB_TUNING]
        self._ap_len = [[int(t * scale), int((t + _STEREO_SPREAD) * scale)]
                        for t in _ALLPASS_TUNING]
        min_delay = min(min(p) for p in self._ap_len)
        self._chunk = max(16, min(max_block, min_delay - 1))

        self._comb_lines = [DelayLine(max(l) + max_block + 8, 2) for l in self._comb_len]
        self._ap_lines = [DelayLine(max(l) + max_block + 8, 2) for l in self._ap_len]
        for dl in self._comb_lines + self._ap_lines:
            dl.prepare(self._chunk + max_block)
        self._comb_damp = [OnePole(1.0) for _ in _COMB_TUNING]
        for f in self._comb_damp:
            f.prepare(sr, 2)
        self._pre = DelayLine(int(sr * 0.25) + max_block + 8, 2)
        self._pre.prepare(max_block)
        self._scratch = np.zeros((2, max_block), dtype=np.float64)

    def reset(self):
        if self._comb_lines is None:
            return
        for dl in self._comb_lines + self._ap_lines:
            dl.reset()
        for f in self._comb_damp:
            f.reset()
        self._pre.reset()

    def _mode_params(self):
        if self.mode == "Room":
            return 0.70, 1.0, 0.0
        if self.mode == "Plate":
            return 0.82, 0.7, 0.15
        if self.mode == "Cathedral":
            return 0.97, 1.3, 0.4
        return 0.90, 1.0, 0.2  # Hall

    def _process(self, x, n):
        base_room, damp_scale, extra_pre = self._mode_params()
        room = base_room * (0.55 + 0.44 * clampf(self.size, 0.0, 1.0))
        room = clampf(room, 0.0, 0.985)
        damp = clampf(self.damping * damp_scale, 0.0, 0.98)

        # Pre-delay separates the source from the tail; without it a big reverb
        # smears the pick attack and everything sounds far away.
        self._pre.write(x)
        pd = int((self.predelay_ms + extra_pre * 20.0) * 1e-3 * self.sr)
        pd = max(pd, n + 1)
        src = self._pre.read_fixed(pd, n) * 0.015

        acc = self._scratch[:, :n]
        acc[...] = 0.0

        # Parallel comb filters -> the tail's density.
        for i, dl in enumerate(self._comb_lines):
            dL = max(self._comb_len[i][0], n + 1)
            dR = max(self._comb_len[i][1], n + 1)
            outL = dl.read_fixed(dL, n)[0]
            outR = dl.read_fixed(dR, n)[1]
            out = np.stack((outL, outR))
            f = self._comb_damp[i]
            # Damping is a one-pole lowpass inside the feedback path: high
            # frequencies die away faster than lows, as in a real room.
            f.b[0] = 1.0 - damp
            f.a[1] = -damp
            filt = f.process(out)
            dl.write(src + filt * room)
            acc += out

        acc *= 0.25

        # Series allpasses -> diffusion. Chunked because their delays are
        # shorter than a block.
        c = self._chunk
        pos = 0
        while pos < n:
            k = min(c, n - pos)
            seg = acc[:, pos:pos + k]
            for i, dl in enumerate(self._ap_lines):
                dL = max(self._ap_len[i][0], k + 1)
                dR = max(self._ap_len[i][1], k + 1)
                bufL = dl.read_fixed(dL, k)[0]
                bufR = dl.read_fixed(dR, k)[1]
                buf = np.stack((bufL, bufR))
                dl.write(seg + buf * 0.5)
                seg = buf - seg
            acc[:, pos:pos + k] = seg
            pos += k

        w = clampf(self.width, 0.0, 1.0)
        mid = (acc[0] + acc[1]) * 0.5
        side = (acc[0] - acc[1]) * 0.5 * w
        wet0 = mid + side
        wet1 = mid - side

        m = clampf(self.mix, 0.0, 1.0)
        out = np.empty((2, n), dtype=np.float64)
        out[0] = x[0] * (1.0 - m * 0.3) + wet0 * m * 3.0
        out[1] = x[1] * (1.0 - m * 0.3) + wet1 * m * 3.0
        return out


# ==========================================================================
# Output section
# ==========================================================================

class Output(Block):
    name = "output"
    stereo = True

    def __init__(self):
        super().__init__()
        self.enabled = True
        self.volume_db = -6.0
        self.width = 1.0
        self.peak_l = 0.0
        self.peak_r = 0.0
        self._g = SmoothedParam(db2lin(-6.0))
        self._limiter_env = OnePole(30.0)

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._g.prepare(max_block)
        self._g.snap(db2lin(self.volume_db))
        self._limiter_env.prepare(sr, 1)
        self._limiter_env.reset()

    def _process(self, x, n):
        w = clampf(self.width, 0.0, 2.0)
        if abs(w - 1.0) > 1e-3:
            mid = (x[0] + x[1]) * 0.5
            side = (x[0] - x[1]) * 0.5 * w
            x = np.stack((mid + side, mid - side))
        self._g.set(db2lin(clampf(self.volume_db, -60.0, 12.0)))
        y = x * self._g.ramp(n)

        # Brick-wall safety limiter. Not a tone control -- it exists so a
        # runaway feedback loop cannot blow up headphones.
        peak = np.max(np.abs(y), axis=0)
        env = self._limiter_env.process(peak)
        over = env > 0.95
        if np.any(over):
            g = np.ones(n)
            g[over] = 0.95 / env[over]
            y = y * g

        self.peak_l = float(np.max(np.abs(y[0]))) if n else 0.0
        self.peak_r = float(np.max(np.abs(y[1]))) if n else 0.0
        return y
