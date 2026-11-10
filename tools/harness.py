"""Offline test harness.

The whole point: a DSP bug in a live path announces itself as "hmm, that
sounds a bit off" three days later. Here it announces itself as a number.

Run:  .venv/Scripts/python.exe tools/harness.py
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amprack.dsp.blocks import AMP_MODELS, CAB_MODELS, DRIVE_MODELS, Delay, Modulation, Reverb  # noqa: E402
from amprack.dsp.chain import Chain  # noqa: E402
from amprack.looper import Looper  # noqa: E402
from amprack.tuner import detect_pitch, freq_to_note  # noqa: E402

SR = 48000.0
FAILURES: list[str] = []
PASSES = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASSES
    if ok:
        PASSES += 1
        print(f"  PASS  {name}" + (f"   {detail}" if detail else ""))
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}   {detail}")


# --------------------------------------------------------------------------
# Signal generators -- deliberately independent of anything in the DSP, so a
# bug in the DSP cannot also corrupt the yardstick measuring it.
# --------------------------------------------------------------------------

def sine(f: float, dur: float, amp: float = 0.3, sr: float = SR) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return amp * np.sin(2 * math.pi * f * t)


def pluck(f: float, dur: float, amp: float = 0.4, sr: float = SR) -> np.ndarray:
    """Harmonic stack with a decaying envelope -- close enough to a guitar note
    to exercise the same frequency range and dynamics."""
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros_like(t)
    for h, w in enumerate((1.0, 0.6, 0.42, 0.25, 0.18, 0.12, 0.08), start=1):
        y += w * np.sin(2 * math.pi * f * h * t + h * 0.7)
    env = np.exp(-t * 3.0) * (1 - np.exp(-t * 400.0))
    return amp * y * env / 2.6


def chord(freqs, dur, amp=0.35, sr=SR) -> np.ndarray:
    out = np.zeros(int(dur * sr))
    for f in freqs:
        out += pluck(f, dur, amp / len(freqs) ** 0.5, sr)
    return out


def run_blocks(chain: Chain, x: np.ndarray, block: int) -> np.ndarray:
    out = np.zeros((2, len(x)))
    for i in range(0, len(x) - block + 1, block):
        out[:, i:i + block] = chain.process(x[i:i + block].copy())
    return out


# --------------------------------------------------------------------------
# 1. Bypass must be transparent, not just quiet
# --------------------------------------------------------------------------

def test_bypass_transparency():
    print("\n[1] bypass transparency")
    x = pluck(196.0, 1.0)

    c = Chain()
    c.prepare(SR, 256)
    c.global_bypass = True
    c.input_gain_db = 0.0
    y = run_blocks(c, x, 256)
    n = (len(x) // 256) * 256
    err = float(np.max(np.abs(y[0, :n] - x[:n])))
    check("global bypass is bit-exact", err == 0.0, f"max abs error {err:.3e}")

    # Every individual block disabled must also be a no-op.
    c2 = Chain()
    c2.prepare(SR, 256)
    for b in c2.blocks.values():
        b.enabled = False
    c2.blocks["output"].enabled = False
    y2 = run_blocks(c2, x, 256)
    err2 = float(np.max(np.abs(y2[0, :n] - x[:n])))
    check("all blocks disabled is bit-exact", err2 == 0.0, f"max abs error {err2:.3e}")


# --------------------------------------------------------------------------
# 2. Nothing produces NaN, inf, or a runaway, on any model
# --------------------------------------------------------------------------

def test_sanity_all_models():
    print("\n[2] numeric sanity across every model")
    rng = np.random.default_rng(7)
    # Hard input: a loud chord over a hot DC-ish offset, plus a burst of noise.
    x = np.concatenate([
        chord([82.41, 123.47, 164.81, 207.65, 246.94, 329.63], 1.2, 0.9),
        rng.standard_normal(int(SR * 0.3)) * 0.8,
        np.zeros(int(SR * 0.5)),
    ])

    worst = 0.0
    bad = []
    for amp in AMP_MODELS:
        for drive in DRIVE_MODELS:
            c = Chain()
            c.prepare(SR, 256)
            c["drive"].enabled = True
            c["drive"].model = drive
            c["drive"].drive = 0.9
            c["amp"].model = amp
            c["amp"].gain = 0.95
            c["gate"].enabled = True
            c["comp"].enabled = True
            for k in ("mod", "delay", "reverb"):
                c[k].enabled = True
            c["delay"].feedback = 0.9
            c["reverb"].size = 1.0
            y = run_blocks(c, x, 256)
            if not np.all(np.isfinite(y)):
                bad.append(f"{amp}/{drive}: non-finite")
            peak = float(np.max(np.abs(y)))
            worst = max(worst, peak)
            if peak > 1.001:
                bad.append(f"{amp}/{drive}: peak {peak:.3f}")
    check("all amp x drive combinations finite and limited", not bad,
          "; ".join(bad[:4]) if bad else f"worst peak {worst:.3f}")

    # Cabs and mods on their own.
    bad = []
    for cab in CAB_MODELS:
        c = Chain()
        c.prepare(SR, 256)
        c["cab"].model = cab
        y = run_blocks(c, x, 256)
        if not np.all(np.isfinite(y)):
            bad.append(cab)
    check("all cabinets finite", not bad, ";".join(bad))

    bad = []
    for mode in Modulation.MODES:
        for rmode in Reverb.MODES:
            c = Chain()
            c.prepare(SR, 256)
            c["mod"].enabled = True
            c["mod"].mode = mode
            c["mod"].feedback = 0.95
            c["reverb"].enabled = True
            c["reverb"].mode = rmode
            c["reverb"].size = 1.0
            c["reverb"].mix = 0.6
            y = run_blocks(c, x, 256)
            if not np.all(np.isfinite(y)):
                bad.append(f"{mode}/{rmode}")
    check("all mod x reverb combinations finite", not bad, ";".join(bad[:4]))


# --------------------------------------------------------------------------
# 3. Block-size invariance
# --------------------------------------------------------------------------

def test_block_size_invariance():
    print("\n[3] block-size invariance (preamp path)")
    x = pluck(110.0, 1.0, 0.5)

    def build():
        c = Chain()
        c["drive"].enabled = True
        c["drive"].drive = 0.6
        c["comp"].enabled = True
        c["gate"].enabled = True
        c["amp"].gain = 0.7
        for k in ("mod", "delay", "reverb"):
            c[k].enabled = False
        return c

    ref = build()
    ref.prepare(SR, 512)
    y_ref = run_blocks(ref, x, 512)

    for block in (64, 128, 256):
        c = build()
        c.prepare(SR, block)
        y = run_blocks(c, x, block)
        n = (len(x) // 512) * 512
        err = float(np.max(np.abs(y[0, :n] - y_ref[0, :n])))
        rel = err / max(float(np.max(np.abs(y_ref[0, :n]))), 1e-12)
        check(f"block {block} matches block 512", rel < 1e-9,
              f"relative max error {rel:.2e}")


# --------------------------------------------------------------------------
# 4. The delay actually delays by what the knob says
# --------------------------------------------------------------------------

def test_delay_time():
    print("\n[4] delay time accuracy")
    for t_ms in (60.0, 250.0, 500.0, 1200.0):
        d = Delay()
        d.enabled = True
        d.mode = "Digital"
        d.time_ms = t_ms
        d.feedback = 0.0
        d.mix = 1.0
        d.tone = 1.0
        d.prepare(SR, 256)
        n_total = int(SR * (t_ms / 1000.0 + 0.4))
        imp = np.zeros((2, n_total))
        imp[:, 10] = 1.0
        out = np.zeros((2, n_total))
        for i in range(0, n_total - 256, 256):
            out[:, i:i + 256] = d.process(imp[:, i:i + 256].copy())
        # Ignore the dry impulse at sample 10; find the echo.
        tail = np.abs(out[0, 200:])
        peak = int(np.argmax(tail)) + 200
        measured = (peak - 10) / SR * 1000.0
        err = abs(measured - t_ms)
        check(f"delay {t_ms:.0f} ms", err < 3.0,
              f"measured {measured:.2f} ms (err {err:.2f} ms)")


# --------------------------------------------------------------------------
# 5. Reverb and delay tails decay instead of growing
# --------------------------------------------------------------------------

def test_tail_stability():
    print("\n[5] feedback stability")
    burst = np.concatenate([pluck(220.0, 0.5, 0.8), np.zeros(int(SR * 12))])

    r = Reverb()
    r.enabled = True
    r.size = 1.0
    r.damping = 0.0
    r.mix = 1.0
    r.mode = "Cathedral"
    r.prepare(SR, 256)
    out = np.zeros((2, len(burst)))
    for i in range(0, len(burst) - 256, 256):
        seg = np.stack((burst[i:i + 256], burst[i:i + 256]))
        out[:, i:i + 256] = r.process(seg)
    # Compare energy in the last second against 4 seconds earlier.
    late = float(np.sqrt(np.mean(out[:, -int(SR):] ** 2)))
    mid = float(np.sqrt(np.mean(out[:, -int(SR * 5):-int(SR * 4)] ** 2)))
    check("reverb tail decays at max size / zero damping",
          np.isfinite(late) and late < mid,
          f"rms {mid:.3e} -> {late:.3e}")

    d = Delay()
    d.enabled = True
    d.feedback = 0.95
    d.mix = 1.0
    d.time_ms = 300.0
    d.mode = "Analog"
    d.prepare(SR, 256)
    out = np.zeros((2, len(burst)))
    for i in range(0, len(burst) - 256, 256):
        seg = np.stack((burst[i:i + 256], burst[i:i + 256]))
        out[:, i:i + 256] = d.process(seg)
    late = float(np.max(np.abs(out[:, -int(SR):])))
    early = float(np.max(np.abs(out[:, int(SR):int(SR * 2)])))
    check("delay at 95% feedback stays bounded",
          np.isfinite(late) and late <= early * 1.05,
          f"peak {early:.3f} -> {late:.3f}")


# --------------------------------------------------------------------------
# 6. Mid-stream events -- the cases a from-silence suite never reaches
# --------------------------------------------------------------------------

def _max_step(y: np.ndarray) -> float:
    return float(np.max(np.abs(np.diff(y))))


def test_midstream_events():
    print("\n[6] mid-stream state (not starting from silence)")

    # (a) a chord struck while an earlier note is still ringing
    lead = pluck(82.41, 2.5, 0.5)
    ch = chord([146.83, 220.0, 293.66], 2.0, 0.5)
    mixed = lead.copy()
    start = int(SR * 0.8)
    mixed[start:start + len(ch)] += ch[:len(mixed) - start]

    c = Chain()
    c.prepare(SR, 256)
    c["drive"].enabled = True
    c["amp"].gain = 0.75
    c["gate"].enabled = True
    c["comp"].enabled = True
    y = run_blocks(c, mixed, 256)
    step = _max_step(y[0])
    check("chord over a ringing note: no discontinuity",
          np.all(np.isfinite(y)) and step < 0.35, f"max sample step {step:.4f}")

    # (b) knob moves with no gap around them -- the zipper-noise case
    x = sine(220.0, 3.0, 0.4)
    c = Chain()
    c.prepare(SR, 256)
    c["amp"].gain = 0.2
    out = np.zeros((2, len(x)))
    block = 256
    for k, i in enumerate(range(0, len(x) - block, block)):
        frac = i / len(x)
        c["amp"].gain = 0.2 + 0.7 * frac
        c["amp"].master = 0.3 + 0.5 * frac
        c["amp"].bass = 0.5 + 0.4 * math.sin(k * 0.05)
        c["amp"].treble = 0.5 + 0.4 * math.cos(k * 0.04)
        c.blocks["output"].volume_db = -12.0 + 8.0 * frac
        out[:, i:i + block] = c.process(x[i:i + block].copy())
    n = (len(x) // block - 1) * block
    step = _max_step(out[0, :n])
    # A 220 Hz sine at full scale steps by at most ~0.03 per sample; anything
    # far above that is a parameter discontinuity, i.e. a click.
    check("continuous knob sweep produces no click",
          np.all(np.isfinite(out)) and step < 0.05, f"max sample step {step:.4f}")

    # (c) loud passage followed by quiet -- checks the compressor and sag
    #     release rather than their attack
    loud = pluck(110.0, 1.5, 1.0)
    quiet = pluck(110.0, 1.5, 0.05)
    x = np.concatenate([loud, quiet])
    c = Chain()
    c.prepare(SR, 256)
    c["comp"].enabled = True
    c["comp"].ratio = 8.0
    c["comp"].threshold_db = -30.0
    c["amp"].sag = 0.9
    y = run_blocks(c, x, 256)
    boundary = len(loud)
    step = _max_step(y[0, boundary - 2000:boundary + 2000])
    check("loud-to-quiet transition is smooth",
          np.all(np.isfinite(y)) and step < 0.2, f"max sample step {step:.4f}")

    # (d) toggling a block mid-note must not click
    x = sine(196.0, 2.0, 0.4)
    c = Chain()
    c.prepare(SR, 256)
    out = np.zeros((2, len(x)))
    for k, i in enumerate(range(0, len(x) - 256, 256)):
        c["mod"].enabled = (k // 20) % 2 == 0
        c["delay"].enabled = (k // 30) % 2 == 0
        out[:, i:i + 256] = c.process(x[i:i + 256].copy())
    check("toggling effects mid-note stays finite",
          np.all(np.isfinite(out)) and float(np.max(np.abs(out))) <= 1.0,
          f"peak {float(np.max(np.abs(out))):.3f}")


# --------------------------------------------------------------------------
# 7. Looper
# --------------------------------------------------------------------------

def test_looper():
    print("\n[7] looper")
    lp = Looper(max_seconds=10.0)
    lp.prepare(SR, 256)
    block = 256

    src = pluck(220.0, 1.0, 0.5)
    src = np.stack((src, src))

    lp.record()
    n_rec = (src.shape[1] // block) * block
    for i in range(0, n_rec, block):
        lp.process(src[:, i:i + block].copy(), block)
    lp.play()
    check("record then play sets a loop length",
          abs(lp.length_seconds - n_rec / SR) < 0.02,
          f"{lp.length_seconds:.3f} s recorded")

    # Silence in -> loop should come back out.
    silence = np.zeros((2, block))
    got = np.zeros((2, lp.length))
    for i in range(0, lp.length - block, block):
        got[:, i:i + block] = lp.process(silence.copy(), block)
    peak = float(np.max(np.abs(got)))
    check("loop plays back", peak > 0.1, f"playback peak {peak:.3f}")

    # Overdub must sum, and undo must put it back.
    lp.pos = 0.0
    before = float(np.sqrt(np.mean(got ** 2)))
    lp.record()  # -> overdub, snapshots for undo
    add = np.stack((pluck(330.0, 0.5, 0.5), pluck(330.0, 0.5, 0.5)))
    for i in range(0, min(add.shape[1], lp.length) - block, block):
        lp.process(add[:, i:i + block].copy(), block)
    lp.play()
    lp.pos = 0.0
    got2 = np.zeros((2, lp.length))
    for i in range(0, lp.length - block, block):
        got2[:, i:i + block] = lp.process(silence.copy(), block)
    after = float(np.sqrt(np.mean(got2 ** 2)))
    check("overdub adds energy", after > before * 1.05,
          f"rms {before:.4f} -> {after:.4f}")

    ok = lp.undo()
    lp.pos = 0.0
    got3 = np.zeros((2, lp.length))
    for i in range(0, lp.length - block, block):
        got3[:, i:i + block] = lp.process(silence.copy(), block)
    undone = float(np.sqrt(np.mean(got3 ** 2)))
    check("undo restores the pre-overdub loop",
          ok and abs(undone - before) / max(before, 1e-9) < 0.02,
          f"rms {after:.4f} -> {undone:.4f} (target {before:.4f})")

    # Half speed / reverse must not blow up or read out of bounds.
    lp.speed = 0.5
    lp.reverse = True
    for _ in range(200):
        lp.process(silence.copy(), block)
    check("reverse + half speed stays finite and in range",
          np.isfinite(lp.pos) and 0 <= lp.pos < lp.length, f"pos {lp.pos:.1f}")

    lp.clear()
    check("clear resets to empty", lp.length == 0 and lp.state == "empty")


# --------------------------------------------------------------------------
# 8. Tuner -- verified against synthesised tones of known frequency
# --------------------------------------------------------------------------

def test_tuner():
    print("\n[8] tuner accuracy")
    worst = 0.0
    bad = []
    for f in (82.41, 110.0, 146.83, 196.0, 246.94, 329.63, 440.0, 659.26, 987.77):
        x = pluck(f, 0.4, 0.4)[-8192:]
        est, clarity = detect_pitch(x, SR)
        if est <= 0:
            bad.append(f"{f:.1f} Hz -> no detection")
            continue
        cents = 1200.0 * math.log2(est / f)
        worst = max(worst, abs(cents))
        if abs(cents) > 5.0:
            bad.append(f"{f:.1f} Hz -> {est:.2f} Hz ({cents:+.1f} c)")
    check("open strings and beyond detected within 5 cents", not bad,
          "; ".join(bad) if bad else f"worst error {worst:.2f} cents")

    # Deliberately harmonic-heavy signal: the octave-error trap.
    t = np.arange(8192) / SR
    f0 = 82.41
    x = (0.2 * np.sin(2 * math.pi * f0 * t) + 1.0 * np.sin(2 * math.pi * 2 * f0 * t)
         + 0.8 * np.sin(2 * math.pi * 3 * f0 * t)) * 0.25
    est, _ = detect_pitch(x, SR)
    cents = 1200.0 * math.log2(est / f0) if est > 0 else 9999
    check("weak fundamental does not cause an octave error", abs(cents) < 10,
          f"{est:.2f} Hz vs {f0:.2f} Hz ({cents:+.1f} c)")

    name, octv, cents = freq_to_note(440.0)
    check("A440 names as A4 at 0 cents",
          name == "A" and octv == 4 and abs(cents) < 0.01, f"{name}{octv} {cents:+.2f}")

    # Silence must report nothing rather than a confident wrong answer.
    est, _ = detect_pitch(np.zeros(8192), SR)
    check("silence reports no pitch", est == 0.0, f"{est}")


# --------------------------------------------------------------------------
# 9. Distortion aliasing -- the reason for the oversampler
# --------------------------------------------------------------------------

def test_aliasing():
    print("\n[9] distortion aliasing")
    # A 2 kHz sine into a high-gain amp. Harmonics land at 4, 6, 8 kHz...
    # Anything appearing at a non-multiple of 2 kHz is folded aliasing.
    x = sine(2000.0, 1.0, 0.4)
    c = Chain()
    c.prepare(SR, 256)
    c["amp"].model = "Metal"
    c["amp"].gain = 0.95
    c["cab"].enabled = False
    y = run_blocks(c, x, 256)[0]
    seg = y[-32768:] * np.hanning(32768)
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(32768, 1 / SR)
    harmonic = np.zeros(len(freqs), dtype=bool)
    for k in range(1, 12):
        harmonic |= np.abs(freqs - 2000.0 * k) < 60.0
    harmonic |= freqs < 120.0
    total = float(np.sum(spec[harmonic] ** 2))
    alias = float(np.sum(spec[~harmonic] ** 2))
    ratio_db = 10 * math.log10(alias / max(total, 1e-20))
    check("aliasing at least 30 dB below the harmonic series", ratio_db < -30.0,
          f"alias/harmonic = {ratio_db:.1f} dB")


# --------------------------------------------------------------------------
# 10. Can it hold the deadline?
# --------------------------------------------------------------------------

def test_cpu_budget():
    print("\n[10] CPU budget")
    for block in (128, 256, 512):
        c = Chain()
        c.prepare(SR, block)
        for k in ("gate", "comp", "drive", "mod", "delay", "reverb"):
            c[k].enabled = True
        x = pluck(110.0, 6.0, 0.5)
        n_blocks = (len(x) // block)
        # Warm up so the first-call overheads do not land in the measurement.
        for i in range(20):
            c.process(x[i * block:(i + 1) * block].copy())
        t0 = time.perf_counter()
        for i in range(n_blocks):
            c.process(x[i * block:(i + 1) * block].copy())
        dt = (time.perf_counter() - t0) / n_blocks
        deadline = block / SR
        load = dt / deadline
        check(f"block {block} ({deadline*1000:.1f} ms deadline) under 50% load",
              load < 0.5, f"{dt*1000:.3f} ms/block = {load*100:.1f}%")


# --------------------------------------------------------------------------
# 11. Broadcast output survives two independent clocks
# --------------------------------------------------------------------------

def test_broadcast_drift():
    print("\n[11] stream-out clock drift")
    from amprack.engine import BroadcastOut

    worst = []
    for ppm in (0, 50, -50, 200, -200, 1000, -1000):
        b = BroadcastOut()
        b.prepare(SR)
        b.running = True
        out = np.zeros((480, 2), dtype=np.float32)
        consumed = 0.0
        produced = 0
        rate = 1.0 + ppm * 1e-6
        n = 256
        phase = np.arange(n)
        for _ in range(int(10 * 60 * SR / n)):      # 10 simulated minutes
            sig = np.sin(2 * np.pi * 440 * (produced + phase) / SR) * 0.3
            b.push(np.stack((sig, sig)), n)
            produced += n
            consumed += n * rate
            while consumed >= 480:
                b._cb(out, 480, None, None)
                consumed -= 480
        fill_ms = ((b._w - b._r) % b._cap) / SR * 1000.0
        if b.underruns or b.overruns or not (40 < fill_ms < 250):
            worst.append(f"{ppm:+d}ppm: under={b.underruns} over={b.overruns} "
                         f"fill={fill_ms:.0f}ms")
    check("stream out holds sync across +/-1000 ppm for 10 minutes", not worst,
          "; ".join(worst) if worst else "no underruns, fill stayed in band")


# --------------------------------------------------------------------------
# 12. Presets and the IR loader -- state that must survive the control path
# --------------------------------------------------------------------------

class _FakeStream:
    """Stands in for a running sounddevice stream: Engine only stops it."""

    def stop(self):
        pass

    def close(self):
        pass


def _write_wav(path: str, data: np.ndarray, sr: int = 48000) -> None:
    import soundfile as sf
    sf.write(path, data.astype(np.float32), sr)


def test_presets_and_ir():
    print("\n[12] presets and IR loader")
    import tempfile

    from amprack import presets as presetlib
    from amprack.engine import Engine

    tmp = tempfile.mkdtemp(prefix="tanin-harness-")

    # (a) a preset that leaves a block out must not inherit the last preset's
    #     settings for it: loading B after A has to sound like loading B.
    a = presetlib.FACTORY["Metal Chug"]      # gate on, drive on, Plate reverb
    b = presetlib.FACTORY["Clean Chime"]     # says nothing about the gate
    after_a = Chain()
    after_a.apply(a)
    after_a.apply(b)
    fresh = Chain()
    fresh.apply(b)
    sa, sf_ = after_a.snapshot(), fresh.snapshot()
    diff = [f"{k}.{nm}" for k in Chain.PARAM_SPEC for nm in Chain.PARAM_SPEC[k]
            if sa[k][nm] != sf_[k][nm]]
    check("preset B after preset A equals preset B alone", not diff,
          "bleeds: " + ", ".join(diff[:6]) if diff else "no carry-over")

    # (b) junk in a preset file must not reach the audio thread as junk.
    c = Chain()
    c.apply({"amp": {"gain": "loud", "master": float("nan")},
             "reverb": {"enabled": "yes please"}, "gate": "not a dict"})
    ok = isinstance(c["amp"].gain, float) and math.isfinite(c["amp"].master) \
        and isinstance(c["reverb"].enabled, bool)
    check("malformed preset values fall back to defaults", ok,
          f"gain={c['amp'].gain!r} master={c['amp'].master!r} "
          f"reverb.enabled={c['reverb'].enabled!r}")

    # (c) a preset or IR change made while audio is stopped must stick.
    e = Engine(os.path.join(tmp, "rec"))
    e.post(lambda: e.chain.apply(presetlib.FACTORY["Ambient Wash"]))
    check("preset picked while stopped is applied",
          e.chain["reverb"].mode == "Cathedral",
          f"reverb mode {e.chain['reverb'].mode!r}")
    #     ...and one queued just before the stream stops is not dropped.
    e.running, e.stream = True, _FakeStream()
    e.post(lambda: e.chain.apply(presetlib.FACTORY["Surf Twang"]))
    e.stop()
    check("command queued as the stream stops still lands",
          e.chain["mod"].mode == "Tremolo" and not e._commands,
          f"mod mode {e.chain['mod'].mode!r}, {len(e._commands)} left queued")

    # (d) presets carry their IR file: loading one loads the IR, loading one
    #     without an IR clears it.
    rng = np.random.default_rng(3)
    ir_a = np.exp(-np.arange(1500) / 300.0) * rng.standard_normal(1500)
    ir_b = np.zeros(64)
    ir_b[0] = 1.0
    path_a = os.path.join(tmp, "cab_a.wav")
    path_b = os.path.join(tmp, "cab_b.wav")
    _write_wav(path_a, ir_a)
    _write_wav(path_b, ir_b)

    e = Engine(os.path.join(tmp, "rec"))
    e.chain.prepare(SR, 256)
    st = Chain().snapshot()
    st["ir"].update(enabled=True, ir_path=path_a)
    err = e.apply_preset(st)
    irb = e.chain["ir"]
    check("preset with an IR file loads that IR",
          not err and irb.enabled and irb.ir_path == path_a and irb.has_ir,
          err or f"enabled={irb.enabled} path={os.path.basename(irb.ir_path)!r}")
    err = e.apply_preset(presetlib.FACTORY["Clean Chime"])
    check("preset without an IR clears the previous one",
          not err and not irb.enabled and irb.ir_path == "" and not irb.has_ir,
          f"enabled={irb.enabled} path={irb.ir_path!r}")
    st["ir"]["ir_path"] = os.path.join(tmp, "missing.wav")
    err = e.apply_preset(st)
    check("missing IR file is reported, not silently half-loaded",
          bool(err) and not irb.enabled and e.chain["amp"].model == st["amp"]["model"],
          err or "no error returned")

    # (e) the IR maths runs on the control thread. Only a reference swap may
    #     reach the audio thread's command queue.
    e = Engine(os.path.join(tmp, "rec"))
    e.chain.prepare(SR, 256)
    e.running, e.stream = True, _FakeStream()     # pretend a stream is up
    calls = {"n": 0}
    real_rfft = np.fft.rfft

    def counting_rfft(*args, **kw):
        calls["n"] += 1
        return real_rfft(*args, **kw)

    err = e.load_ir(path_a)
    np.fft.rfft = counting_rfft
    try:
        while e._commands:
            e._commands.popleft()()
    finally:
        np.fft.rfft = real_rfft
    e.running, e.stream = False, None
    check("IR load does no FFT on the audio thread",
          not err and calls["n"] == 0 and e.chain["ir"].has_ir,
          err or f"{calls['n']} rfft call(s) in the drained command")

    # (f) level: two very different IRs land at a similar loudness, and
    #     neither a trim move nor an IR swap steps the output.
    def ir_rms(path):
        en = Engine(os.path.join(tmp, "rec"))
        en.chain.prepare(SR, 256)
        en.load_ir(path)
        blk = en.chain["ir"]
        noise = np.random.default_rng(5).standard_normal(int(SR)) * 0.1
        out = np.concatenate([blk.process(noise[i:i + 256].copy())
                              for i in range(0, len(noise) - 255, 256)])
        return float(np.sqrt(np.mean(out[int(SR * 0.2):] ** 2)))

    ra, rb = ir_rms(path_a), ir_rms(path_b)
    gap = abs(20 * math.log10(ra / rb))
    check("different IRs load at a matched level", gap < 1.5,
          f"level gap {gap:.1f} dB")

    e = Engine(os.path.join(tmp, "rec"))
    e.chain.prepare(SR, 256)
    e.load_ir(path_b)
    blk = e.chain["ir"]
    x = sine(220.0, 1.5, 0.3)
    out = np.zeros(len(x))
    for k, i in enumerate(range(0, len(x) - 255, 256)):
        if k == 60:
            blk.gain_db = 12.0                   # trim jumps 12 dB in one go
        if k == 120:
            e.load_ir(path_a)                    # swap the cab mid-note
        out[i:i + 256] = blk.process(x[i:i + 256].copy())
    n = (len(x) // 256) * 256
    step = _max_step(out[:n])
    # The yardstick is the sine's own slope in the loudest steady stretch on
    # either side of the swap; a click is a step well above that.
    natural = max(_max_step(out[70 * 256:118 * 256]), _max_step(out[130 * 256:n]))
    check("IR trim move and IR swap do not click",
          np.all(np.isfinite(out)) and step <= natural * 1.25,
          f"max sample step {step:.4f} vs steady {natural:.4f}")

    # (g) preset names end up as file names: refuse anything that could
    #     escape the folder or that Windows will not create.
    bad = ["", "   ", "../evil", "a/b", "a\\b", "CON", "nul", "com1.json",
           "x:y", "what?", "name.", "tab\there", "x" * 100]
    good = ["Clean Chime", "Lead #2", "طنين نظيف", "Ambient (wet)"]
    slipped = [repr(nm) for nm in bad if not presetlib.validate_name(nm)]
    refused = [repr(nm) for nm in good if presetlib.validate_name(nm)]
    check("unsafe preset names are refused, normal ones accepted",
          not slipped and not refused,
          f"accepted {slipped} refused {refused}" if slipped or refused
          else f"{len(bad)} refused, {len(good)} accepted")
    pdir = os.path.join(tmp, "presets")
    try:
        presetlib.save_preset(pdir, "../escaped", {"amp": {}})
        raised = False
    except ValueError:
        raised = True
    check("save_preset will not write outside the preset folder",
          raised and not os.path.exists(os.path.join(tmp, "escaped.json")),
          "raised ValueError" if raised else "wrote it")


def main():
    print("=" * 68)
    print("Tanin offline harness")
    print("=" * 68)
    t0 = time.perf_counter()
    test_bypass_transparency()
    test_sanity_all_models()
    test_block_size_invariance()
    test_delay_time()
    test_tail_stability()
    test_midstream_events()
    test_looper()
    test_tuner()
    test_aliasing()
    test_cpu_budget()
    test_broadcast_drift()
    test_presets_and_ir()
    dt = time.perf_counter() - t0

    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{PASSES} passed, {len(FAILURES)} FAILED  ({dt:.1f}s)")
        for f in FAILURES:
            print("  x " + f)
        return 1
    print(f"{PASSES} passed, 0 failed  ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
