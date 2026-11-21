"""Tests for the instrument suite: engine foundation (phase 1) and vocals with
auto-tune (phase 2).

Hooked into tools/harness.py the same way as dsp_tests.py: the harness passes
in its ``check`` so results land in the same tally.
"""

from __future__ import annotations

import math
import tempfile

import numpy as np

from amprack.dsp.autotune import AutoTune
from amprack.dsp.chain import Chain
from amprack.dsp.scales import NOTE_NAMES, SCALES, allowed_notes, pitch_class_mask, snap
from amprack.tuner import detect_pitch

SR = 48000.0


# -- signal helpers (independent of the DSP under test) ---------------------

def _voice(f, dur, amp=0.2, vib=0.0, formant_hz=None):
    """A sung-vowel stand-in: a harmonic stack, optionally with vibrato and a
    fixed formant peak (so formant preservation can be measured)."""
    t = np.arange(int(dur * SR)) / SR
    inst = f * (1.0 + vib * np.sin(2 * math.pi * 5.5 * t))
    ph = 2 * math.pi * np.cumsum(inst) / SR
    y = np.zeros_like(t)
    for h in range(1, 30):
        fh = f * h
        if fh > SR * 0.45:
            break
        if formant_hz:
            w = math.exp(-((fh - formant_hz) / 350.0) ** 2) + 0.05 / h
        else:
            w = (1.0, .5, .35, .2, .12, .08)[h - 1] if h <= 6 else 0.0
        y += w * np.sin(h * ph)
    return amp * y / max(np.max(np.abs(y)), 1e-9)


def _run(block, x, b=256):
    out = np.zeros_like(x)
    for i in range(0, len(x) - b + 1, b):
        out[i:i + b] = block.process(x[i:i + b].copy())
    return out


def _pitch(y, start_s=0.5, n=8192):
    seg = y[int(start_s * SR):int(start_s * SR) + n]
    f, _ = detect_pitch(seg, SR, fmin=40.0)
    return f


def _midi(f):
    return 69.0 + 12.0 * math.log2(f / 440.0) if f > 0 else float("nan")


def _engine(n=256, channels=2):
    from amprack.engine import Engine
    e = Engine(tempfile.mkdtemp(prefix="tanin-suite-"))
    e.samplerate = SR
    e.active_blocksize = n
    e._prepare_buffers(channels)
    return e


def _feed(e, x0, x1, n=256):
    """Run the real callback over two input channels; returns (2, len) out."""
    L = min(len(x0), len(x1))
    out = np.zeros((2, L))
    ind = np.zeros((n, 2), dtype=np.float32)
    od = np.zeros((n, 2), dtype=np.float32)
    for i in range(0, L - n + 1, n):
        ind[:, 0] = x0[i:i + n]
        ind[:, 1] = x1[i:i + n]
        e._callback(ind, od, n, None, None)
        out[:, i:i + n] = od.T
    return out


# ==========================================================================
# Phase 1: engine foundation
# ==========================================================================

def test_engine_foundation(check):
    print("\n[S1] engine foundation: instruments, mixer, input pickers")
    n = 256
    x = 0.3 * np.sin(2 * math.pi * 110 * np.arange(int(SR)) / SR)
    silent = np.zeros_like(x)

    # Guitar through the engine is bit-for-bit the guitar chain on its own.
    e = _engine(n)
    ref = Chain()
    ref.prepare(SR, n)
    for c in (e.chain, ref):
        for k in ("gate", "comp", "drive", "delay", "reverb"):
            c[k].enabled = True
    got = _feed(e, x, silent, n)
    want = np.zeros((2, len(x)))
    for i in range(0, len(x) - n + 1, n):
        want[:, i:i + n] = ref.process(x[i:i + n].astype(np.float32).astype(np.float64))
    want = np.clip(want, -1, 1).astype(np.float32).astype(np.float64)
    err = float(np.max(np.abs(got - want)))
    check("guitar through instrument + mixer is bit-exact to the old chain",
          err == 0.0 and e.callback_errors == 0,
          f"max error {err:.3e}, errors {e.callback_errors} {e.last_callback_exc}")

    # An instrument that is off is never called.
    calls = {"n": 0}
    real = e.vocals.process

    def counting(*a):
        calls["n"] += 1
        return real(*a)
    e.vocals.process = counting
    e.vocals.enabled = False
    _feed(e, x[:n * 8], x[:n * 8], n)
    off_calls = calls["n"]
    e.vocals.enabled = True
    _feed(e, x[:n * 8], x[:n * 8], n)
    check("an instrument that is off costs no CPU (never processed)",
          off_calls == 0 and calls["n"] == 8, f"off: {off_calls} calls, on: {calls['n']}")
    e.vocals.process = real

    # Input picker: guitar on In 1, vocals on In 2; each hears only its jack.
    e = _engine(n)
    e.vocals.enabled = True
    e.chain.global_bypass = True
    e.vocals.chain.global_bypass = True
    _feed(e, silent[:n * 20], x[:n * 20], n)
    g_pk, v_pk = e.mixer["guitar"].peak, e.mixer["vocals"].peak
    e.guitar.input_channel, e.vocals.input_channel = 1, 0
    _feed(e, silent[:n * 20], x[:n * 20], n)
    g2, v2 = e.mixer["guitar"].peak, e.mixer["vocals"].peak
    check("each instrument reads its own input (and can be swapped)",
          g_pk == 0.0 and v_pk > 0.29 and g2 > 0.29 and v2 == 0.0,
          f"In2 signal: guitar {g_pk:.2f} vocals {v_pk:.2f}; swapped: "
          f"guitar {g2:.2f} vocals {v2:.2f}")

    # Mono interface: an instrument pointed at a missing jack falls back.
    e1 = _engine(n, channels=1)
    check("on a 1-input interface every instrument falls back to input 1",
          e1.vocals.input_channel == 0 and e1.guitar.input_channel == 0,
          f"guitar {e1.guitar.input_channel}, vocals {e1.vocals.input_channel}")

    # Send matrix: vocals only to Stream, guitar only to Phones.
    e = _engine(n)
    e.vocals.enabled = True
    e.chain.global_bypass = True
    e.vocals.chain.global_bypass = True
    e.mixer["vocals"].sends.update(phones=0.0, record=0.0, stream=1.0)
    e.mixer["guitar"].sends.update(phones=1.0, record=1.0, stream=0.0)
    tone_v = 0.25 * np.sin(2 * math.pi * 440 * np.arange(n * 20) / SR)
    _feed(e, silent[:n * 20], tone_v, n)
    ph = float(np.max(np.abs(e.mixer.bus_view("phones", n))))
    st = float(np.max(np.abs(e.mixer.bus_view("stream", n))))
    check("send matrix routes vocals to Stream only",
          ph == 0.0 and st > 0.2, f"phones {ph:.3f}, stream {st:.3f}")

    # Solo, mute and pan.
    e = _engine(n)
    e.vocals.enabled = True
    e.chain.global_bypass = True
    e.vocals.chain.global_bypass = True
    e.mixer["vocals"].solo = True
    _feed(e, x[:n * 20], silent[:n * 20], n)
    solo_out = float(np.max(np.abs(e.mixer.bus_view("phones", n))))
    e.mixer["vocals"].solo = False
    e.mixer["guitar"].pan = -1.0
    out = _feed(e, x[:n * 20], silent[:n * 20], n)
    check("solo silences the others; hard-left pan empties the right side",
          solo_out == 0.0 and float(np.max(np.abs(out[1, n * 4:]))) == 0.0
          and float(np.max(np.abs(out[0, n * 4:]))) > 0.29,
          f"solo leak {solo_out:.3f}, right {np.max(np.abs(out[1, n*4:])):.3f}")

    # A fader move glides instead of stepping.
    e = _engine(n)
    e.chain.global_bypass = True
    dc = np.full(n * 10, 0.5)
    out_a = _feed(e, dc, silent[:n * 10], n)
    e.mixer["guitar"].volume_db = -20.0
    out_b = _feed(e, dc[:n * 2], silent[:n * 2], n)
    seq = np.concatenate([out_a[0], out_b[0]])
    step = float(np.max(np.abs(np.diff(seq))))
    check("fader moves are smoothed (no click)", step < 0.01,
          f"largest sample step {step:.4f}")


# ==========================================================================
# Phase 2: auto-tune
# ==========================================================================

def test_scales(check):
    print("\n[S2] keys and scales")
    ok = all(pitch_class_mask(k, s).sum() == (len(SCALES[s]) if s != "Custom" else 12)
             for k in range(12) for s in SCALES)
    check("all 12 keys x all scales build", ok and len(SCALES) == 12, f"{len(SCALES)} scales")
    notes = allowed_notes(pitch_class_mask(9, "Natural Minor"), 24, 84)   # A minor
    bad = [m for m in notes if (int(m) % 12) not in (9, 11, 0, 2, 4, 5, 7)]
    check("A minor allows only A B C D E F G, C1..C6",
          not bad and notes.min() >= 24 and notes.max() <= 84,
          f"{len(notes)} notes, {notes.min():.0f}..{notes.max():.0f}")
    check("snap picks the nearest scale note",
          snap(60.9, allowed_notes(pitch_class_mask(0, "Major"), 24, 84)) == 60.0
          and snap(61.1, allowed_notes(pitch_class_mask(0, "Major"), 24, 84)) == 62.0)


def test_autotune_snaps(check):
    print("\n[S3] auto-tune snaps to key")
    # Off-key notes in different keys and scales, hard tune.
    cases = [  # (input Hz, key, scale) -> expected MIDI
        (233.0, 2, "Major", 57),          # Bb3-ish in D major -> A3
        (300.0, 0, "Chromatic", 62),      # between D4 and D#4 (closer to D)
        (180.0, 4, "Minor Pentatonic", 55),   # E minor pent: G3
        (420.0, 7, "Mixolydian", 69),     # G mixolydian: G#4+40c -> A4 (no G#)
        (140.0, 9, "Harmonic Minor", 49),  # A harmonic minor: C#3? no -> see below
    ]
    # Last case: 140 Hz is MIDI 49.4; A harmonic minor has G# (pc 8) and A
    # (pc 9) but not C# (pc 1), so the nearest allowed note is 48 (C3) or 50
    # (D3). 49.4 is nearer to 50.
    cases[-1] = (140.0, 9, "Harmonic Minor", 50)
    bad = []
    for f, key, scale, want in cases:
        at = AutoTune()
        at.retune_ms, at.key, at.scale, at.low_note = 0.0, key, scale, 24
        at.prepare(SR, 256)
        got = _midi(_pitch(_run(at, _voice(f, 1.0))))
        if not abs(got - want) < 0.05:
            bad.append(f"{f}Hz {NOTE_NAMES[key]} {scale}: {got:.2f} want {want}")
    check("off-key notes land on the scale note (within 5 cents)", not bad,
          "; ".join(bad) or f"{len(cases)} cases")

    # Every key, major scale: a note 40 cents sharp of the 3rd lands on the 3rd.
    bad = []
    for key in range(12):
        target = 48 + key + 4
        f = 440.0 * 2 ** ((target + 0.4 - 69) / 12)
        at = AutoTune()
        at.retune_ms, at.key, at.scale = 0.0, key, "Major"
        at.prepare(SR, 256)
        got = _midi(_pitch(_run(at, _voice(f, 0.9))))
        if abs(got - target) > 0.05:
            bad.append(f"{NOTE_NAMES[key]}: {got:.2f}")
    check("all 12 keys correct a sharp note to the scale", not bad, "; ".join(bad))

    # Octave range: singing below the range snaps to the lowest allowed note.
    at = AutoTune()
    at.retune_ms, at.low_note, at.high_note = 0.0, 48, 72   # C3..C5
    at.prepare(SR, 256)
    got = _midi(_pitch(_run(at, _voice(110.0, 1.0))))       # A2 = 45
    check("octave range: below the low limit snaps up to it", abs(got - 48) < 0.05,
          f"A2 in -> {got:.2f} (C3 = 48)")

    # Octave shift, with and without formant keep.
    bad = []
    for octv in (-2, -1, 1, 2):
        for formant in (True, False):
            at = AutoTune()
            at.retune_ms, at.octave, at.formant, at.low_note = 0.0, octv, formant, 24
            at.prepare(SR, 256)
            got = _midi(_pitch(_run(at, _voice(220.0, 1.2)), 0.7))
            want = 57 + 12 * octv
            if abs(got - want) > 0.05:
                bad.append(f"oct {octv:+d} formant {formant}: {got:.2f} want {want}")
    check("octave shift -2..+2 lands on the right octave", not bad, "; ".join(bad))


def _centroid(y):
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return float(np.sum(sp * f) / np.sum(sp))


def test_autotune_character(check):
    print("\n[S4] auto-tune character: formants, retune speed, latency, noise")
    x = _voice(180.0, 1.4, formant_hz=900.0)
    res = {}
    for formant in (True, False):
        at = AutoTune()
        at.retune_ms, at.octave, at.formant, at.low_note = 0.0, 1, formant, 24
        at.prepare(SR, 256)
        y = _run(at, x)
        res[formant] = _centroid(y[int(0.6 * SR):int(1.2 * SR)])
    c_in = _centroid(x[int(0.6 * SR):int(1.2 * SR)])
    check("formant keep holds the vowel colour an octave up",
          abs(res[True] / c_in - 1) < 0.25 and res[False] / c_in > 1.6,
          f"spectral centre: in {c_in:.0f} Hz, keep {res[True]:.0f} Hz, "
          f"off {res[False]:.0f} Hz")

    # Retune speed: hard tune corrects at once, 400 ms is still gliding.
    def corr_after(ms, secs):
        at = AutoTune()
        at.retune_ms = ms
        at.prepare(SR, 256)
        sig = _voice(440.0 * 2 ** (0.45 / 12), secs + 0.05)   # 45 cents sharp
        for i in range(0, int(secs * SR) - 255, 256):
            at.process(sig[i:i + 256].copy())
        return at.correction_cents
    hard = corr_after(0.0, 0.1)
    slow = corr_after(400.0, 0.1)
    check("retune speed: 0 ms snaps at once, 400 ms glides",
          abs(hard + 45) < 3 and -20 < slow < -3,
          f"after 0.1 s: hard {hard:.1f} cents, slow {slow:.1f} cents (full = -45)")

    # Fixed latency: with mix 0 the output is the input, 12.5 ms late, exactly.
    at = AutoTune()
    at.mix = 0.0
    at.prepare(SR, 256)
    y = _run(at, x)
    d = at.latency
    err = float(np.max(np.abs(y[d:len(x) - 256] - x[:len(x) - 256 - d])))
    check("latency is a fixed 12.5 ms", d == 600 and err == 0.0,
          f"{d} samples ({d / SR * 1000:.1f} ms), max error {err:.1e}")

    # Breath / consonant noise passes through at the same level.
    rng = np.random.default_rng(3)
    noise = 0.05 * rng.standard_normal(int(SR))
    at = AutoTune()
    at.prepare(SR, 256)
    y = _run(at, noise)
    lvl = 20 * math.log10(np.std(y[int(0.2 * SR):]) / np.std(noise[int(0.2 * SR):]))
    check("unvoiced sounds pass through at the same level", abs(lvl) < 1.0
          and at.target_note == -1.0, f"level change {lvl:+.2f} dB")

    # Natural setting keeps vibrato; hard tune flattens it.
    def vib_depth(ms):
        at = AutoTune()
        at.retune_ms = ms
        at.prepare(SR, 256)
        y = _run(at, _voice(220.0, 2.0, vib=0.012))
        fs = [_midi(detect_pitch(y[i:i + 2048], SR)[0])
              for i in range(int(0.5 * SR), int(1.9 * SR), 512)]
        return float(np.nanmax(fs) - np.nanmin(fs))
    hard_v, nat_v = vib_depth(0.0), vib_depth(400.0)
    check("slow retune keeps vibrato, hard tune removes it",
          nat_v > 0.25 and hard_v < 0.6 * nat_v,
          f"pitch swing: hard {hard_v * 100:.0f} cents, natural {nat_v * 100:.0f} cents")

    # Piano mode hook: held notes become the only targets.
    at = AutoTune()
    at.retune_ms = 0.0
    mask = np.zeros(12, dtype=bool)
    mask[7] = True            # only G
    at.external_mask = mask
    at.prepare(SR, 256)
    got = _midi(_pitch(_run(at, _voice(220.0, 1.0))))
    check("piano-mode targets override the scale", abs(got - 55) < 0.05,
          f"A3 in, only G allowed -> {got:.2f} (G3 = 55)")


def test_autotune_realtime(check):
    print("\n[S5] auto-tune real-time rules")
    x = _voice(207.0, 1.0, vib=0.01)

    def run(b):
        at = AutoTune()
        at.retune_ms = 30.0
        at.prepare(SR, 512)
        return _run(at, x, b)
    a, b = run(64), run(512)
    L = (len(x) // 512) * 512
    err = float(np.max(np.abs(a[:L] - b[:L])))
    check("auto-tune output identical at 64 and 512 buffers", err == 0.0,
          f"max error {err:.1e}")

    at = AutoTune()
    at.prepare(SR, 256)
    at.enabled = False
    blk = x[:256].copy()
    check("auto-tune bypass is bit-exact", np.array_equal(at.process(blk), blk))


# ==========================================================================
# Phase 2: the vocal chain and presets
# ==========================================================================

def test_vocal_chain(check):
    print("\n[S6] vocal chain and presets")
    from amprack.instruments.vocals import (
        VOCAL_PRESETS, VocalChain, VocalsInstrument, apply_vocal_preset)

    x = _voice(196.0, 1.0)
    c = VocalChain()
    c.prepare(SR, 256)
    c.global_bypass = True
    y = np.concatenate([c.process(x[i:i + 256].copy())[0] for i in range(0, 48000 - 255, 256)])
    ok_global = np.array_equal(y, x[:len(y)])
    c.global_bypass = False
    for b in c.blocks.values():
        b.enabled = False
    y = np.concatenate([c.process(x[i:i + 256].copy())[0] for i in range(0, 48000 - 255, 256)])
    check("vocal chain bypass and all-off are bit-exact",
          ok_global and np.array_equal(y, x[:len(y)]))

    def run(b, preset):
        v = VocalsInstrument()
        apply_vocal_preset(v, preset)
        v.prepare(SR, 512)
        out = np.zeros((2, len(x)))
        for i in range(0, len(x) - b + 1, b):
            out[:, i:i + b] = v.process(x[i:i + b].copy(), b)
        return out
    worst = []
    for name in VOCAL_PRESETS:
        a, b = run(64, name), run(512, name)
        L = (len(x) // 512) * 512
        rel = float(np.max(np.abs(a[:, :L] - b[:, :L])) / (np.max(np.abs(b)) + 1e-12))
        if not (rel < 1e-9 and np.all(np.isfinite(a)) and np.max(np.abs(a)) <= 1.0):
            worst.append(f"{name} {rel:.1e}")
    check("every vocal preset: finite, limited, same at 64 and 512 buffers",
          not worst, "; ".join(worst) or f"{len(VOCAL_PRESETS)} presets")

    v = VocalsInstrument()
    at = v.autotune
    at.key, at.scale, at.low_note = 5, "Blues", 40
    apply_vocal_preset(v, "Hard Tune")
    kept = (at.key, at.scale, at.low_note) == (5, "Blues", 40) and at.retune_ms == 0.0
    v.chain.apply({"autotune": {"retune_ms": "fast", "key": float("nan"),
                                "custom": [1, 2]}, "comp": "junk"})
    sane = (isinstance(at.retune_ms, float) and at.key == 0
            and len(at.custom) == 12 and v.chain["comp"].ratio == 3.0)
    check("presets keep your key; bad preset values fall back to defaults",
          kept and sane, f"kept={kept}, sane={sane}")

    names = set(VOCAL_PRESETS)
    check("the five planned vocal presets exist",
          {"Natural Polish", "Hard Tune", "Radio Voice", "Big Choir",
           "Chat Voice"} <= names, ", ".join(sorted(names)))


def test_suite_cpu(check):
    print("\n[S7] CPU with three instruments (guitar + vocals + piano stand-in)")
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench_instruments import measure
    r = measure(256, seconds=3.0)
    check("three instruments at 256 samples under 60% load",
          r["mean"] < 0.6 and r["errors"] == 0,
          f"mean {r['mean']*100:.1f}%, p99 {r['p99']*100:.1f}% of "
          f"{r['deadline_ms']:.1f} ms")


def run_all(check):
    test_engine_foundation(check)
    test_scales(check)
    test_autotune_snaps(check)
    test_autotune_character(check)
    test_autotune_realtime(check)
    test_vocal_chain(check)
    test_suite_cpu(check)
