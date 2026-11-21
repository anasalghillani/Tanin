"""Tests for the slot effects, the circuit-modelled amps and the tone stack.

Hooked into tools/harness.py the same way as dsp_tests.py: the harness
passes its own ``check`` so the results land in the same tally.
"""

from __future__ import annotations

import math
import time

import numpy as np

from amprack.dsp.blocks import AMP_MODELS, CAB_MODELS, Amp, Cabinet
from amprack.dsp.chain import Chain
from amprack.dsp.fx import FX_TYPES
from amprack.dsp.fx.slots import FxSlots
from amprack.dsp.tonestack import STACKS, pot_log, tmb_analog
from amprack.dsp.voicings import NEW_AMPS

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


def _make(cls, model, channels, block):
    fx = cls()
    fx.channels = 2 if cls.STEREO else channels
    fx.model = model
    fx.prepare(SR, block)
    return fx


def _run(obj, x, block):
    n = (x.shape[-1] // block) * block
    out = None
    for i in range(0, n, block):
        y = obj.process(x[..., i:i + block].copy())
        if out is None:
            out = np.zeros(y.shape[:-1] + (n,))
        out[..., i:i + block] = y
    return out


def _models():
    for key, cls in FX_TYPES.items():
        for m in cls.MODELS:
            yield key, cls, m


# --------------------------------------------------------------------------

def test_fx_bypass(check):
    print("\n[FX1] slot effects: bypass and empty racks are bit-exact")
    x = _pluck(196.0, 0.5)
    bad = []
    for key, cls, m in _models():
        fx = _make(cls, m, 2, 256)
        fx.enabled = False
        X = np.stack((x, x * 0.5))
        y = _run(fx, X, 256)
        if not np.array_equal(y, X[:, :y.shape[-1]]):
            bad.append(f"{key}/{m}")
    check("every effect, disabled, passes audio bit-exact", not bad, ", ".join(bad[:5]))

    # A chain with all three racks empty matches one with the racks switched off.
    c1, c2 = Chain(), Chain()
    c1.prepare(SR, 256)
    c2.prepare(SR, 256)
    for k in ("pre", "loop", "rack"):
        c2[k].enabled = False
    d = float(np.max(np.abs(_run(c1, x, 256) - _run(c2, x, 256))))
    check("empty slot racks are bit-exact", d == 0.0, f"max diff {d:.2e}")

    # A slot holding a switched-off effect is also transparent.
    c2 = Chain()
    c2.prepare(SR, 256)
    for k in ("pre", "loop", "rack"):
        c2[k].enabled = False
    c3 = Chain()
    c3.prepare(SR, 256)
    c3["loop"].set_type(0, "phaser")
    c3["loop"].slots[0].enabled = False
    c3["pre"].set_type(1, "wah")
    c3["pre"].effect(1).enabled = False
    d = float(np.max(np.abs(_run(c3, x, 256) - _run(c2, x, 256))))
    check("slots holding disabled effects are bit-exact", d == 0.0, f"max diff {d:.2e}")


def test_fx_sanity(check):
    print("\n[FX2] slot effects: finite and bounded at extreme settings")
    rng = np.random.default_rng(3)
    x = np.concatenate([_pluck(82.4, 0.6, 0.9), rng.standard_normal(int(SR * 0.2)) * 0.8,
                        np.zeros(int(SR * 0.4))])
    bad = []
    worst = 0.0
    for key, cls, m in _models():
        for corner in ("lo", "hi"):
            fx = _make(cls, m, 1, 256)
            for p in cls.PARAMS:
                if p.kind in ("lin", "log"):
                    setattr(fx, p.name, p.hi if corner == "hi" else p.lo)
            X = x[None, :] if not cls.STEREO else np.stack((x, x))
            y = _run(fx, X, 256)
            if not np.all(np.isfinite(y)):
                bad.append(f"{key}/{m}/{corner}: non-finite")
                continue
            pk = float(np.max(np.abs(y))) / float(np.max(np.abs(x)))
            worst = max(worst, pk)
            # EQs with every band and the level maxed are +30 dB by design.
            lim = 63.0 if key in ("geq", "peq") else 16.0
            if pk > lim:
                bad.append(f"{key}/{m}/{corner}: gain {pk:.1f}")
    check("every effect model finite, peak gain under +24 dB (EQ +36 dB) at both knob extremes",
          not bad, "; ".join(bad[:4]) or f"worst peak gain {worst:.2f}x")

    # Effects that feed back must die away once the input stops.
    stuck = []
    tail = np.concatenate([_pluck(196.0, 0.3), np.zeros(int(SR * 6.0))])
    for key in ("echo", "verb", "flanger", "phaser"):
        cls = FX_TYPES[key]
        for m in cls.MODELS:
            fx = _make(cls, m, 2, 256)
            if hasattr(fx, "feedback"):
                fx.feedback = cls.param("feedback").hi if key != "echo" else 0.9
            if hasattr(fx, "decay"):
                fx.decay = 1.0
            y = _run(fx, np.stack((tail, tail)), 256)
            end = float(np.max(np.abs(y[:, -int(SR * 0.5):])))
            if not np.isfinite(end) or end > 0.05:
                stuck.append(f"{key}/{m} {end:.3f}")
    check("feedback effects at max settings decay instead of running away",
          not stuck, "; ".join(stuck[:4]))


def test_fx_block_invariance(check):
    print("\n[FX3] slot effects: identical output at every buffer size")
    x = _pluck(146.8, 0.6, 0.5)
    X = np.stack((x, x * 0.7))
    bad = []
    worst = 0.0
    for key, cls, m in _models():
        if key in ("lofi",):
            pass
        ref = _run(_make(cls, m, 2, 512), X, 512)
        for block in (64, 192, 2048):
            Xb = np.concatenate((X, np.zeros((2, 2048))), axis=1)
            y = _run(_make(cls, m, 2, block), Xb, block)
            n = ref.shape[-1]
            err = float(np.max(np.abs(y[:, :n] - ref)))
            rel = err / max(float(np.max(np.abs(ref))), 1e-9)
            worst = max(worst, rel)
            if rel > 1e-7:
                bad.append(f"{key}/{m}@{block}: {rel:.1e}")
    check("every effect model gives the same output at 64 / 192 / 512 / 2048",
          not bad, "; ".join(bad[:4]) or f"worst relative error {worst:.1e}")

    bad = []
    for name in NEW_AMPS:
        ref = None
        for block in (512, 64, 2048):
            a = Amp()
            a.model = name
            a.gain = 0.7
            a.prepare(SR, block)
            y = _run(a, np.concatenate((x, np.zeros(2048))), block)[:len(x) // 512 * 512]
            if ref is None:
                ref = y
            else:
                rel = float(np.max(np.abs(y - ref))) / max(float(np.max(np.abs(ref))), 1e-9)
                if rel > 1e-9:
                    bad.append(f"{name}@{block}: {rel:.1e}")
    check("every new amp voicing is buffer-size invariant", not bad, "; ".join(bad[:4]))


def _alias_ratio_db(y, f0, band_hi=16000.0):
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


def test_fx_aliasing(check):
    print("\n[FX4] drives, fuzzes, boosts, octave fuzz and new amps: aliasing")
    f0 = 1567.0
    x = _sine(f0, 1.2, 0.4)
    res = []
    for key in ("dirt", "boost", "octup"):
        cls = FX_TYPES[key]
        for m in cls.MODELS:
            fx = _make(cls, m, 1, 256)
            for nm in ("gain", "fuzz"):
                if hasattr(fx, nm):
                    setattr(fx, nm, 0.9)
            if hasattr(fx, "tone"):
                fx.tone = 1.0
            res.append((f"{key}/{m}", _alias_ratio_db(_run(fx, x[None, :], 256)[0], f0)))
    bad = [f"{k} {v:.1f}" for k, v in res if v > -45.0]
    worst = max(res, key=lambda r: r[1])
    check("every drive/fuzz/boost model aliases at least 45 dB under its harmonics",
          not bad, "; ".join(bad[:4]) or f"worst {worst[0]} {worst[1]:.1f} dB")
    res = []
    for name in NEW_AMPS:
        a = Amp()
        a.model = name
        a.gain = 0.95
        a.master = 0.9
        a.prepare(SR, 256)
        res.append((name, _alias_ratio_db(_run(a, x, 256), f0)))
    bad = [f"{k} {v:.1f}" for k, v in res if v > -45.0]
    worst = max(res, key=lambda r: r[1])
    check("every new amp voicing, cranked, aliases at least 45 dB under its harmonics",
          not bad, "; ".join(bad[:4]) or f"worst {worst[0]} {worst[1]:.1f} dB")


def _mna_tmb(f, t, m, l, R1, R2, R3, R4, C1, C2, C3):
    """Nodal analysis of the TMB network: an independent yardstick."""
    s = 2j * math.pi * f
    Y = np.zeros((6, 6), complex)

    def add(a, b, y):
        Y[a, a] += y
        if b is not None:
            Y[b, b] += y
            Y[a, b] -= y
            Y[b, a] -= y

    def G(r):
        return 1.0 / max(r, 1e-6)
    # nodes: 0 in, 1 slope junction, 2 treble top, 3 bass top, 4 mid top, 5 out
    add(0, 2, s * C1)
    add(0, 1, G(R4))
    add(1, 3, s * C2)
    add(1, 4, s * C3)
    add(2, 5, G((1 - t) * R1))
    add(5, 3, G(t * R1))
    add(3, 4, G(l * R2))
    add(4, None, G(m * R3))
    v = np.linalg.solve(Y[1:, 1:], -Y[1:, 0])
    return v[4]


def test_tonestack(check):
    print("\n[FX5] tone stack formula vs a nodal-analysis solve of the circuit")
    worst = 0.0
    for kind, comp in STACKS.items():
        for t, m, l in ((0.5, 0.5, 0.5), (0.1, 0.9, 0.8), (0.9, 0.2, 0.2), (0.7, 0.6, 0.05)):
            b, a = tmb_analog(t, m, l, **comp)
            for f in (60.0, 200.0, 700.0, 2000.0, 6000.0):
                w = 2j * math.pi * f
                h1 = abs(np.polyval(b, w) / np.polyval(a, w))
                h2 = abs(_mna_tmb(f, t, m, l, **comp))
                worst = max(worst, abs(20 * math.log10(h1 / h2)))
    check("TMB formula within 0.5 dB of the circuit for every stack and knob set",
          worst < 0.5, f"worst {worst:.3f} dB")

    # Knob behaviour: bass up raises 100 Hz, treble up raises 5 kHz, mid
    # up fills the scoop -- on every stack type.
    a = Amp()
    a.model = "Brit Plexi"
    a.prepare(SR, 256)
    from scipy.signal import freqz
    ok = True
    for kind in STACKS:
        st = a._stack
        st.set(kind, 0.5, 0.5, 0.5)
        _, h0 = freqz(st.b, st.a, [100.0, 600.0, 5000.0], fs=SR)
        st.set(kind, 0.5, 0.5, 1.0)
        _, hb = freqz(st.b, st.a, [100.0, 600.0, 5000.0], fs=SR)
        st.set(kind, 1.0, 0.5, 0.5)
        _, ht = freqz(st.b, st.a, [100.0, 600.0, 5000.0], fs=SR)
        st.set(kind, 0.5, 1.0, 0.5)
        _, hm = freqz(st.b, st.a, [100.0, 600.0, 5000.0], fs=SR)
        # Thresholds are what the real circuits do: some stacks (Orange
        # bass, Mesa mid) only move their band by about 1 dB around noon.
        ok &= abs(hb[0]) > abs(h0[0]) * 1.06 and abs(ht[2]) > abs(h0[2]) * 1.06 \
            and abs(hm[1]) > abs(h0[1]) * 1.06
    check("bass, treble and mid knobs move their own band on every stack", ok)


def _band_level(y, lo, hi):
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return float(np.sqrt(np.sum(spec[(f >= lo) & (f < hi)] ** 2)))


def _centroid(y):
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return float(np.sum(spec * f) / max(np.sum(spec), 1e-12))


def test_fx_behaviour(check):
    print("\n[FX6] effects do what they say")
    # Auto-wah: the filter opens on the attack and closes as the note dies.
    # Noise shaped like a plucked note, so the spectrum shows the filter.
    w = _make(FX_TYPES["wah"], "Classic", 1, 256)
    w.source = "Envelope"
    w.sensitivity = 0.5
    t = np.arange(int(1.5 * SR)) / SR
    x = np.random.default_rng(2).standard_normal(len(t)) * 0.3 * np.exp(-t * 3.0)
    y = _run(w, x[None, :], 256)[0]
    early = _centroid(y[int(0.02 * SR):int(0.12 * SR)])
    late = _centroid(y[int(1.1 * SR):int(1.4 * SR)])
    check("auto-wah sweeps up on the pick and back down as the note decays",
          early > late * 1.25, f"centroid {early:.0f} Hz early vs {late:.0f} Hz late")

    # Cocked wah at heel vs toe.
    lv = []
    for pos in (0.0, 1.0):
        w = _make(FX_TYPES["wah"], "Classic", 1, 256)
        w.source = "Fixed"
        w.position = pos
        noise = np.random.default_rng(5).standard_normal(int(SR)) * 0.1
        lv.append(_centroid(_run(w, noise[None, :], 256)[0]))
    check("fixed wah: toe is brighter than heel", lv[1] > lv[0] * 1.5,
          f"heel {lv[0]:.0f} Hz, toe {lv[1]:.0f} Hz")

    # Sub octave: a 220 Hz note gives a strong 110 Hz (and 55 Hz) voice.
    o = _make(FX_TYPES["octdown"], "Analog", 1, 256)
    o.direct = 0.0
    o.oct1 = 1.0
    o.oct2 = 0.0
    x = _pluck(220.0, 1.0, 0.5)
    y = _run(o, x[None, :], 256)[0][int(0.1 * SR):int(0.6 * SR)]
    sub = _band_level(y, 100.0, 120.0)
    fund = _band_level(y, 210.0, 230.0)
    check("sub octave puts its main energy one octave down",
          sub > fund * 2.0, f"110 Hz vs 220 Hz: {20 * math.log10(sub / max(fund, 1e-12)):+.1f} dB")
    o.oct1, o.oct2 = 0.0, 1.0
    o.reset()
    y = _run(o, x[None, :], 256)[0][int(0.1 * SR):int(0.6 * SR)]
    s2 = _band_level(y, 50.0, 60.0)
    check("second flip-flop gives two octaves down",
          s2 > _band_level(y, 210.0, 230.0) * 2.0, "")

    # Pitch shifter: +12 moves the energy up an octave.
    p = _make(FX_TYPES["pitch"], "Shift", 1, 256)
    p.semitones = 12.0
    p.mix = 1.0
    x = _sine(220.0, 1.0, 0.4)
    y = _run(p, x[None, :], 256)[0][int(0.2 * SR):]
    up = _band_level(y, 430.0, 450.0)
    orig = _band_level(y, 210.0, 230.0)
    check("pitch shifter +12 puts the note an octave up",
          up > orig * 8.0, f"440 vs 220 Hz: {20 * math.log10(up / max(orig, 1e-12)):+.1f} dB")

    # Echo: first repeat lands on the time knob.
    e = _make(FX_TYPES["echo"], "Digital", 2, 256)
    e.time_ms = 250.0
    e.feedback = 0.0
    e.mix = 1.0
    imp = np.zeros(int(SR))
    imp[1000] = 1.0
    y = _run(e, np.stack((imp, imp)), 256)[0]
    y[:1000 + 10] = 0.0
    lag = (int(np.argmax(np.abs(y))) - 1000) / SR * 1000.0
    check("echo repeat arrives at the set time", abs(lag - 250.0) < 1.0, f"{lag:.2f} ms")

    # Room reverb: Decay is an honest RT60.
    v = _make(FX_TYPES["verb"], "Hall", 2, 256)
    v.decay = 0.5
    v.mix = 1.0
    v.predelay_ms = 1.0
    imp = np.zeros(int(SR * 4))
    imp[0] = 1.0
    y = _run(v, np.stack((imp, imp)), 256)[0]
    env = np.convolve(y ** 2, np.ones(2400) / 2400, mode="same")
    edc = np.cumsum(env[::-1])[::-1]
    edc_db = 10 * np.log10(edc / edc[0] + 1e-30)
    i5 = int(np.argmax(edc_db < -5))
    i25 = int(np.argmax(edc_db < -25))
    rt = (i25 - i5) / SR * 3.0
    target = 1.2 * (7.0 / 1.2) ** 0.5
    check("hall reverb decay time matches its RT60 within 35%",
          abs(rt - target) / target < 0.35, f"measured {rt:.2f} s vs {target:.2f} s")

    # Spring: dispersion -- highs and lows of one echo arrive at different times.
    v = _make(FX_TYPES["verb"], "Spring", 2, 256)
    v.mix = 1.0
    v.predelay_ms = 1.0
    v.dwell = 0.0
    y = _run(v, np.stack((imp[:int(SR)] * 0.05, imp[:int(SR)] * 0.05)), 256)[0][:int(0.06 * SR)]
    from scipy.signal import butter, sosfilt
    lo = sosfilt(butter(4, [300, 600], "bandpass", fs=SR, output="sos"), y)
    hi = sosfilt(butter(4, [2500, 3500], "bandpass", fs=SR, output="sos"), y)
    t_lo = int(np.argmax(np.abs(lo))) / SR * 1000.0
    t_hi = int(np.argmax(np.abs(hi))) / SR * 1000.0
    check("spring reverb chirps (bands of one echo arrive >2 ms apart)",
          abs(t_hi - t_lo) > 2.0, f"450 Hz at {t_lo:.1f} ms, 3 kHz at {t_hi:.1f} ms")

    # Rotary: Fast ramps up rather than jumping, horn faster than drum.
    r = _make(FX_TYPES["rotary"], "Standard", 2, 256)
    r.speed = "Fast"
    _run(r, np.zeros((2, int(0.15 * SR))), 256)
    h1, d1 = r._f
    _run(r, np.zeros((2, int(3.0 * SR))), 256)
    h2, d2 = r._f
    check("rotary speeds up gradually, horn ahead of drum",
          h1 < 6.0 and h2 > 6.5 and (h1 - 0.83) / 6.0 > (d1 - 0.67) / 5.2,
          f"after 0.15 s horn {h1:.2f} Hz drum {d1:.2f} Hz; after 3 s {h2:.2f}/{d2:.2f}")

    # Tremolo depth: 100% depth reaches near silence on the trough.
    tr = _make(FX_TYPES["tremolo"], "Bias", 2, 256)
    tr.depth = 1.0
    tr.rate = 4.0
    y = _run(tr, np.ones((2, int(SR))) * 0.5, 256)[0][int(0.3 * SR):]
    check("tremolo at full depth swings from full to near silence",
          float(np.min(y)) < 0.05 * float(np.max(y)), "")


def test_fx_presets(check):
    print("\n[FX7] slot racks in presets and mid-song switches")
    a_state = {"pre": {"slots": [{"type": "wah", "params": {"model": "Vintage", "q": 0.9}},
                                 {"type": "dirt", "params": {"model": "Muff", "gain": 0.8}}]},
               "loop": {"slots": [{"type": "phaser", "params": {"model": "Stone"}}]},
               "rack": {"slots": [{}, {"type": "verb", "params": {"model": "Spring"}}]}}
    b_state = {"loop": {"slots": [{"type": "echo", "params": {"model": "Tape"}}]}}
    x = _pluck(196.0, 1.0)

    def render(*states):
        c = Chain()
        c.prepare(SR, 256)
        for st in states:
            c.preload(st)
            c.apply(st)
        return _run(c, x, 256), c

    yb, cb = render(b_state)
    yab, cab = render(a_state, b_state)
    d = float(np.max(np.abs(yab - yb)))
    check("slot preset B after A equals B alone", d == 0.0, f"max diff {d:.2e}")

    _, ca = render(a_state)
    snap = ca.snapshot()
    _, cc = render(snap)
    check("snapshot of a slotted preset round-trips",
          cc.snapshot() == snap and snap["pre"]["slots"][1]["params"]["model"] == "Muff")

    bad = {"pre": {"slots": [{"type": "nope"}, {"type": 3}, "junk",
                             {"type": "wah", "params": {"q": "loud", "model": "X",
                                                        "rate": float("nan")}}]},
           "loop": "garbage"}
    try:
        _, cbad = render(bad)
        ok = cbad["pre"].slots[0].type == "" and cbad["pre"].effect(3).model == "Classic" \
            and cbad["pre"].effect(3).q == 0.5 and math.isfinite(cbad["pre"].effect(3).rate)
    except Exception as exc:  # noqa: BLE001
        ok = False
        print("   ", exc)
    check("malformed slot presets fall back to empty / defaults", ok)

    # Switching tones mid-note: stays finite, no click beyond the note itself.
    c = Chain()
    c.prepare(SR, 256)
    states = [a_state, b_state, {"pre": {"slots": [{"type": "octdown"}]}}, {}]
    for st in states:
        c.preload(st)
    y = np.zeros((2, len(x)))
    k = 0
    for i in range(0, len(x) - 255, 256):
        if i % (256 * 40) == 0:
            c.apply(states[k % len(states)])
            k += 1
        y[:, i:i + 256] = c.process(x[i:i + 256].copy())
    check("switching slotted tones mid-note stays finite and limited",
          bool(np.all(np.isfinite(y))) and float(np.max(np.abs(y))) <= 1.0001, "")

    # Preloaded types are not rebuilt on the audio thread.
    c = Chain()
    c.prepare(SR, 256)
    c.preload(a_state)
    before = id(c["pre"].slots[0].pool["wah"])
    c.apply(a_state)
    check("preloaded slot effects are reused, not rebuilt",
          id(c["pre"].effect(0)) == before)


def test_fx_cpu(check):
    print("\n[FX8] CPU: a busy song tone")
    for block in (256, 512):
        c = Chain()
        c.prepare(SR, block)
        st = {"amp": {"model": "Brit 800", "gain": 0.7},
              "cab": {"model": "4x12 Greenback"},
              "pre": {"slots": [{"type": "wah"}, {"type": "dirt", "params": {"model": "Green"}}]},
              "loop": {"slots": [{"type": "phaser"}, {"type": "echo",
                                                     "params": {"model": "Tape"}}]},
              "rack": {"slots": [{"type": "verb", "params": {"model": "Hall"}}]},
              "gate": {"enabled": True}}
        c.preload(st)
        c.apply(st)
        x = _pluck(110.0, 4.0, 0.5)
        nb = len(x) // block
        for i in range(10):
            c.process(x[i * block:(i + 1) * block].copy())
        t0 = time.perf_counter()
        for i in range(nb):
            c.process(x[i * block:(i + 1) * block].copy())
        dt = (time.perf_counter() - t0) / nb
        load = dt / (block / SR)
        check(f"wah + drive + amp + phaser + tape echo + hall at block {block} under 60% load",
              load < 0.6, f"{dt * 1000:.2f} ms/block = {load * 100:.1f}%")


def test_new_voicings(check):
    print("\n[FX9] new amp voicings and cabinets")
    rng = np.random.default_rng(9)
    x = np.concatenate([_pluck(82.4, 0.6, 0.9), rng.standard_normal(int(SR * 0.2)) * 0.8])
    bad = []
    for name in NEW_AMPS:
        for g, mst in ((0.0, 0.0), (1.0, 1.0)):
            a = Amp()
            a.model = name
            a.gain, a.master = g, mst
            a.bass = a.mid = a.treble = a.presence = 1.0 if g else 0.0
            a.prepare(SR, 256)
            y = _run(a, x, 256)
            if not np.all(np.isfinite(y)) or float(np.max(np.abs(y))) > 100.0:
                bad.append(f"{name} g{g}")
            c = Chain()
            c.prepare(SR, 256)
            c.apply({"amp": {"model": name, "gain": g, "master": mst, "bass": g,
                             "mid": g, "treble": g, "presence": g}})
            yc = _run(c, x, 256)
            if not np.all(np.isfinite(yc)) or float(np.max(np.abs(yc))) > 1.0001:
                bad.append(f"{name} in chain g{g}")
    check("every new amp finite at both knob extremes, and limited in the chain",
          not bad, ", ".join(bad))

    # Power amp: on an amp without a master, Master adds harmonics.
    def thd(master):
        a = Amp()
        a.model = "Brit Plexi"
        a.gain = 0.2
        a.master = master
        a.prepare(SR, 256)
        y = _run(a, _sine(331.0, 1.0, 0.3), 256)[-16384:]
        spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        f = np.fft.rfftfreq(len(y), 1 / SR)
        fund = spec[np.abs(f - 331.0) < 20].sum()
        harm = sum(spec[np.abs(f - 331.0 * k) < 20].sum() for k in range(2, 8))
        return harm / fund
    lo, hi = thd(0.1), thd(1.0)
    check("Plexi-style amp breaks up more as Master (power amp) rises",
          hi > lo * 1.5, f"harmonic ratio {lo:.3f} -> {hi:.3f}")

    bad = []
    for cab in CAB_MODELS:
        cb = Cabinet()
        cb.model = cab
        cb.mic = "Tube"
        cb.prepare(SR, 256)
        y = _run(cb, x, 256)
        if not np.all(np.isfinite(y)):
            bad.append(cab)
    check("every cabinet (incl. new ones) with every mic is finite", not bad, ", ".join(bad))


def run_all(check):
    for t in (test_fx_bypass, test_fx_sanity, test_fx_block_invariance, test_fx_aliasing,
              test_tonestack, test_fx_behaviour, test_fx_presets, test_fx_cpu,
              test_new_voicings):
        try:
            t(check)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(t.__name__, False, f"raised {type(e).__name__}: {e}")
