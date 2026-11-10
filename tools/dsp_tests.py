"""DSP regression tests for the 2026-11 audit items (4, 5, 6, 7, 12, 13, 18).

Kept in their own file so tools/harness.py only needs a two-line hook; the
harness passes in its own ``check`` so results land in the same tally.
Each test was written to fail against the pre-fix DSP first.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import hilbert

from amprack.dsp.blocks import (
    AMP_MODELS,
    Amp,
    Drive,
    Modulation,
    NoiseGate,
    Output,
    Reverb,
)
from amprack.dsp.chain import Chain

SR = 48000.0


def _sine(f, dur, amp=0.3):
    t = np.arange(int(dur * SR)) / SR
    return amp * np.sin(2 * math.pi * f * t)


def _pluck(f, dur, amp=0.4):
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(t)
    for h, w in enumerate((1.0, 0.6, 0.42, 0.25, 0.18, 0.12, 0.08), start=1):
        y += w * np.sin(2 * math.pi * f * h * t + h * 0.7)
    env = np.exp(-t * 3.0) * (1 - np.exp(-t * 400.0))
    return amp * y * env / 2.6


def _run(block_obj, x, block):
    """Feed a block object in fixed-size pieces. x is (n,) or (2, n)."""
    n = (x.shape[-1] // block) * block
    out = None
    for i in range(0, n, block):
        y = block_obj.process(x[..., i:i + block].copy())
        if out is None:
            out = np.zeros(y.shape[:-1] + (n,))
        out[..., i:i + block] = y
    return out


# --------------------------------------------------------------------------
# Item 4: reverb pre-delay and comb lengths must not depend on buffer size
# --------------------------------------------------------------------------

def test_reverb_buffer_independence(check):
    print("\n[A4] reverb pre-delay and combs vs buffer size")
    n_total = 2048 * 12
    imp = np.zeros((2, n_total))
    imp[:, 0] = 1.0

    def render(block):
        r = Reverb()
        r.enabled = True
        r.mode = "Room"          # extra_pre = 0, so the knob is the whole pre-delay
        r.predelay_ms = 15.0
        r.mix = 1.0
        r.prepare(SR, block)
        return _run(r, imp, block)

    outs = {b: render(b) for b in (64, 256, 2048)}
    # Expected onset: dry impulse at 0, then the first comb echo at
    # predelay + shortest comb length.
    pd = int(15.0e-3 * SR)
    first_comb = int(1116 * SR / 44100.0)
    expect = pd + first_comb
    onsets = {}
    for b, y in outs.items():
        tail = np.abs(y[0, 1:])
        nz = np.nonzero(tail > 1e-9)[0]
        onsets[b] = int(nz[0]) + 1 if len(nz) else -1
    check("reverb onset = pre-delay + first comb at every buffer size",
          all(o == expect for o in onsets.values()),
          f"expected {expect}, got " + ", ".join(f"{b}:{o}" for b, o in onsets.items()))

    ref = outs[64]
    worst = 0.0
    for b in (256, 2048):
        err = float(np.max(np.abs(outs[b] - ref)))
        worst = max(worst, err / max(float(np.max(np.abs(ref))), 1e-12))
    check("reverb impulse response identical at 64 / 256 / 2048", worst < 1e-9,
          f"relative max error {worst:.2e}")


# --------------------------------------------------------------------------
# Item 5: the Drive bias knob-in-the-table must actually change the sound
# --------------------------------------------------------------------------

def _harmonic_db(y, f0, k):
    seg = y[-32768:] * np.hanning(32768)
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(32768, 1 / SR)

    def level(f):
        band = np.abs(freqs - f) < 25.0
        return float(np.sqrt(np.sum(spec[band] ** 2)))

    return 20 * math.log10(max(level(k * f0), 1e-12) / max(level(f0), 1e-12))


def test_drive_bias(check):
    print("\n[A5] drive bias produces even harmonics")
    f0 = 331.0
    x = _sine(f0, 1.0, 0.05)
    res = {}
    for model in ("Screamer", "Rat", "Muff", "Fuzz"):
        d = Drive()
        d.enabled = True
        d.model = model
        d.drive = 0.3
        d.tone = 1.0
        d.prepare(SR, 256)
        y = _run(d, x, 256)
        res[model] = _harmonic_db(y, f0, 2)
    detail = ", ".join(f"{m} H2 {v:.1f} dB" for m, v in res.items())
    biased_ok = all(res[m] > -40.0 for m in ("Rat", "Muff", "Fuzz"))
    check("biased models (Rat/Muff/Fuzz) have 2nd harmonic above -40 dB",
          biased_ok, detail)
    check("unbiased Screamer stays symmetric (2nd harmonic below -60 dB)",
          res["Screamer"] < -60.0, detail)


# --------------------------------------------------------------------------
# Item 6: aliasing, measured with a tone that does NOT divide the sample rate
# --------------------------------------------------------------------------

def _alias_ratio_db(y, f0, band_hi=16000.0):
    """Energy at non-harmonic frequencies below band_hi, relative to the
    harmonic series. f0 is chosen so folded harmonics land between real ones."""
    seg = y[-32768:] * np.blackman(32768)
    spec = np.abs(np.fft.rfft(seg)) ** 2
    freqs = np.fft.rfftfreq(32768, 1 / SR)
    inband = (freqs > 40.0) & (freqs < band_hi)
    harmonic = np.zeros(len(freqs), dtype=bool)
    for k in range(1, int(SR / 2 / f0) + 2):
        harmonic |= np.abs(freqs - f0 * k) < 15.0
    tot = float(np.sum(spec[inband & harmonic]))
    ali = float(np.sum(spec[inband & ~harmonic]))
    return 10 * math.log10(max(ali, 1e-30) / max(tot, 1e-30))


def test_aliasing_real(check):
    print("\n[A6] distortion aliasing (non-commensurate tone)")
    f0 = 1567.0   # G6-ish; 48000 / 1567 is not an integer, so aliases are visible
    x = _sine(f0, 1.2, 0.4)
    worst = []
    for model in ("Crunch", "Metal"):
        c = Chain()
        c.prepare(SR, 256)
        c["amp"].model = model
        c["amp"].gain = 0.95
        c["cab"].enabled = False
        y = _run(c, x, 256)[0]
        worst.append((model, _alias_ratio_db(y, f0)))
    d = Drive()
    d.enabled = True
    d.model = "Muff"
    d.drive = 0.9
    d.tone = 1.0
    d.prepare(SR, 256)
    worst.append(("Muff pedal", _alias_ratio_db(_run(d, x, 256), f0)))
    detail = ", ".join(f"{m} {v:.1f} dB" for m, v in worst)
    check("aliasing below 16 kHz at least 45 dB under the harmonics",
          all(v < -45.0 for _, v in worst), detail)


# --------------------------------------------------------------------------
# Item 7: the safety limiter must catch the very first loud sample
# --------------------------------------------------------------------------

def test_limiter(check):
    print("\n[A7] output safety limiter")
    x = np.concatenate([_sine(440.0, 0.3, 0.2), _sine(440.0, 0.3, 3.0)])
    x = np.stack((x, x))
    o = Output()
    o.volume_db = 0.0
    o.prepare(SR, 256)
    y = _run(o, x, 256)
    peak = float(np.max(np.abs(y)))
    check("a sudden 3.0 peak is held at the 0.95 ceiling from sample one",
          peak <= 0.9501, f"output peak {peak:.3f}")

    quiet = np.stack((_sine(440.0, 0.3, 0.5),) * 2)
    o2 = Output()
    o2.volume_db = 0.0
    o2.prepare(SR, 256)
    yq = _run(o2, quiet, 256)
    err = float(np.max(np.abs(yq - quiet[:, :yq.shape[1]])))
    check("signal under the ceiling passes untouched", err == 0.0, f"max error {err:.2e}")

    # Recovery: after the loud burst, gain comes back within ~0.3 s.
    x3 = np.concatenate([_sine(440.0, 0.2, 3.0), _sine(440.0, 0.6, 0.3)])
    x3 = np.stack((x3, x3))
    o3 = Output()
    o3.volume_db = 0.0
    o3.prepare(SR, 256)
    y3 = _run(o3, x3, 256)
    tail_gain = float(np.max(np.abs(y3[0, -4800:]))) / 0.3
    check("limiter releases after the burst", tail_gain > 0.99, f"gain {tail_gain:.3f}")


# --------------------------------------------------------------------------
# Item 12: inter-stage coupling filters inside the amp are actually used
# --------------------------------------------------------------------------

def _band_db(y, lo, hi):
    seg = y[-32768:] * np.hanning(32768)
    spec = np.abs(np.fft.rfft(seg)) ** 2
    freqs = np.fft.rfftfreq(32768, 1 / SR)
    band = (freqs >= lo) & (freqs < hi)
    return 10 * math.log10(max(float(np.sum(spec[band])), 1e-30) /
                           max(float(np.sum(spec)), 1e-30))


def test_amp_interstage(check):
    print("\n[A12] amp inter-stage filters")
    if not hasattr(Amp, "interstage"):
        check("Amp has an inter-stage filter switch", False, "Amp.interstage missing")
        return
    # Low E plus a sub-sonic thump: the case coupling caps exist for.
    t = np.arange(int(SR * 1.2)) / SR
    x = 0.25 * np.sin(2 * math.pi * 82.41 * t) + 0.25 * np.sin(2 * math.pi * 25.0 * t)
    res = {}
    for on in (False, True):
        a = Amp()
        a.interstage = on
        a.model = "Metal"
        a.gain = 0.9
        a.prepare(SR, 256)
        y = _run(a, x, 256)
        res[on] = (_band_db(y, 4000.0, 24000.0), _band_db(y, 0.0, 60.0))
    fizz_drop = res[False][0] - res[True][0]
    mud_drop = res[False][1] - res[True][1]
    check("inter-stage lowpass trims fizz above 4 kHz by at least 1.5 dB",
          fizz_drop >= 1.5, f"{fizz_drop:+.1f} dB")
    check("inter-stage highpass trims sub-60 Hz mud by at least 1.5 dB",
          mud_drop >= 1.5, f"{mud_drop:+.1f} dB")
    check("inter-stage filters are on by default", Amp().interstage is True)


# --------------------------------------------------------------------------
# Item 13: noise gate must not flutter on a note decaying through threshold
# --------------------------------------------------------------------------

def test_gate_flutter(check):
    print("\n[A13] noise gate flutter")
    rng = np.random.default_rng(3)
    dur = 3.0
    t = np.arange(int(SR * dur)) / SR
    # Low E decaying from -20 dBFS to about -80 dBFS, over a -66 dBFS hiss floor.
    amp = 0.1 * np.exp(-t * 2.3)
    x = amp * np.sin(2 * math.pi * 82.41 * t) + rng.standard_normal(len(t)) * 5e-4

    g = NoiseGate()
    g.enabled = True
    g.threshold_db = -45.0
    g.prepare(SR, 256)
    y = _run(g, x, 256)
    n = len(y)
    # Gain as seen through the analytic envelope (amplitude of a sine is exact).
    env_in = np.abs(hilbert(x[:n]))
    env_out = np.abs(hilbert(y))
    gain = env_out / np.maximum(env_in, 1e-9)
    # Look only where the note is still clearly above the hiss.
    keep = amp[:n] > 0.004
    gdb = 20 * np.log10(np.maximum(gain[keep], 1e-6))
    # Downsample to 5 ms steps and count re-openings (gain rising >1 dB
    # after it has started closing).
    step = int(SR * 0.005)
    g5 = gdb[::step]
    lowest = g5[0]
    reopen = 0
    for v in g5[1:]:
        if v > lowest + 1.0 and lowest < -1.0:
            reopen += 1
            lowest = v
        lowest = min(lowest, v)
    # Ripple at the note's own frequency while open/closing.
    ripple = float(np.max(np.abs(np.diff(g5))))
    check("gate closes once without re-opening on a decaying note", reopen == 0,
          f"{reopen} re-openings")
    check("gate gain has no audio-rate ripple (<3 dB per 5 ms)", ripple < 3.0,
          f"worst 5 ms step {ripple:.1f} dB")


# --------------------------------------------------------------------------
# Item 18: vibrato latency, and chorus/vibrato buffer-size independence
# --------------------------------------------------------------------------

def test_vibrato_latency(check):
    print("\n[A18] vibrato latency")
    lat = {}
    for block in (64, 256, 2048):
        m = Modulation()
        m.enabled = True
        m.mode = "Vibrato"
        m.depth = 0.5
        m.rate = 0.0
        m.prepare(SR, block)
        imp = np.zeros((2, block * 4 if block > 512 else 4096))
        imp[:, 10] = 1.0
        y = _run(m, imp, block)
        lat[block] = (int(np.argmax(np.abs(y[0]))) - 10) / SR * 1000.0
    detail = ", ".join(f"{b}:{v:.2f} ms" for b, v in lat.items())
    check("vibrato adds at most 5 ms of delay at depth 0.5",
          all(v <= 5.0 for v in lat.values()), detail)
    check("vibrato delay is the same at every buffer size",
          max(lat.values()) - min(lat.values()) < 0.05, detail)

    x = np.stack((_pluck(196.0, 1.0),) * 2)
    outs = {}
    for block in (64, 2048):
        m = Modulation()
        m.enabled = True
        m.mode = "Chorus"
        m.prepare(SR, block)
        outs[block] = _run(m, x, block)
    n = min(o.shape[1] for o in outs.values())
    err = float(np.max(np.abs(outs[64][:, :n] - outs[2048][:, :n])))
    check("chorus output identical at 64 and 2048 buffers", err < 1e-9,
          f"max error {err:.2e}")


def run_all(check):
    for t in (test_reverb_buffer_independence, test_drive_bias, test_aliasing_real,
              test_limiter, test_amp_interstage, test_gate_flutter,
              test_vibrato_latency):
        try:
            t(check)
        except Exception as e:  # a crash is a failure, not an abort
            check(t.__name__, False, f"raised {type(e).__name__}: {e}")
