"""Tests for the mic sidechain and the Talkbox slot effect.

Hooked into tools/harness.py the same way as fx_tests.py. The voice is
synthetic (a pulse train through fixed vowel formants), so the tests run
anywhere and know exactly which vowel is sung when.
"""

from __future__ import annotations

import math
import tempfile
import time

import numpy as np
from scipy.signal import lfilter

from amprack.dsp.fx import FX_TYPES, sidechain
from amprack.dsp.fx.sidechain import Sidechain

SR = 48000.0
VOWELS = {"a": (730, 1090, 2440), "i": (270, 2290, 3010), "u": (300, 870, 2240)}


def _voice(seq="aiu", seg=0.8, gap=0.2, f0=120.0, amp=0.1):
    """Vowels from seq, seg seconds each, with gap seconds of silence between."""
    n_seg, n_gap = int(seg * SR), int(gap * SR)
    out = []
    for v in seq:
        ex = np.zeros(n_seg)
        ex[::int(SR / f0)] = 1.0
        y = ex
        for f in VOWELS[v]:
            r = math.exp(-math.pi * 80 / SR)
            w = 2 * math.pi * f / SR
            y = lfilter([1 - r], [1, -2 * r * math.cos(w), r * r], y)
        out += [y / np.max(np.abs(y)) * amp, np.zeros(n_gap)]
    return np.concatenate(out)


def _guitar(n, amp=0.4):
    t = np.arange(n) / SR
    g = np.zeros(n)
    for f in (82.4, 123.5, 164.8):
        g += np.sign(np.sin(2 * math.pi * f * t)) * 0.33
    return amp * g


def _make(model, channels=1, block=256, sc=None):
    fx = FX_TYPES["talkbox"]()
    fx.channels = channels
    fx.model = model
    if sc is not None:
        fx.sidechain = sc
    fx.prepare(SR, block)
    return fx


def _run(fx, sc, mic, gtr, block, channels=1):
    n = (len(gtr) // block) * block
    out = np.zeros((channels, n))
    for i in range(0, n, block):
        sc.write(mic[i:i + block], block)
        out[:, i:i + block] = fx.process(np.stack([gtr[i:i + block]] * channels))
    return out


def _ratio(y, s, e):
    """2-3 kHz over 0.6-1.2 kHz in [s, e) seconds: high for 'ee', low for 'ah'/'oo'."""
    seg = y[int(s * SR):int(e * SR)]
    seg = seg[:4096] * np.hanning(min(4096, len(seg)))
    S = np.abs(np.fft.rfft(seg))
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    return S[(f > 2000) & (f < 3100)].sum() / (S[(f > 600) & (f < 1200)].sum() + 1e-12)


def test_talkbox_follows_mouth(check):
    print("\n[TB1] talkbox: the guitar takes the mic's vowel")
    mic = _voice()
    g = _guitar(len(mic))
    for model in FX_TYPES["talkbox"].MODELS:
        sc = Sidechain()
        sc.prepare(256)
        y = _run(_make(model, sc=sc), sc, mic, g, 256)[0]
        ra, ri, ru = (_ratio(y, k * 1.0 + 0.3, k * 1.0 + 0.7) for k in range(3))
        dry = _ratio(g, 0.3, 0.7)
        check(f"{model}: 'ee' is brighter than 'ah' and 'oo' by 4x+ (dry guitar has no vowel)",
              ri > 4 * ra and ri > 4 * ru,
              f"ah {ra:.2f} ee {ri:.2f} oo {ru:.2f} dry {dry:.2f}")


def test_talkbox_gate(check):
    print("\n[TB2] talkbox: the mouth closes when you stop talking")
    mic = _voice("a", seg=0.8, gap=0.8)
    g = _guitar(len(mic))
    for model in FX_TYPES["talkbox"].MODELS:
        sc = Sidechain()
        sc.prepare(256)
        fx = _make(model, sc=sc)
        y = _run(fx, sc, mic, g, 256)[0]
        talk = float(np.sqrt(np.mean(y[int(0.3 * SR):int(0.7 * SR)] ** 2)))
        quiet = float(np.sqrt(np.mean(y[int(1.2 * SR):int(1.5 * SR)] ** 2)))
        check(f"{model}: closed mouth (Muffled) is at least 8 dB under an open one",
              quiet < talk * 0.4, f"open {talk:.3f} closed {quiet:.3f}")
    # No mic at all (offline render, no interface input): Dry passes the guitar.
    fx = _make("Vowel", sc=Sidechain())
    fx.closed = "Dry"
    x = _guitar(4096)[None, :]
    y = fx.process(x.copy()[:, :256])
    check("no mic + Closed = Dry: guitar passes unchanged",
          float(np.max(np.abs(y - x[:, :256]))) < 1e-12, "")


def test_talkbox_invariance(check):
    print("\n[TB3] talkbox: identical at every buffer size, with a mic")
    mic = _voice("aiu", seg=0.3, gap=0.1)[:49152]
    g = _guitar(len(mic))
    for model in FX_TYPES["talkbox"].MODELS:
        sc = Sidechain()
        sc.prepare(2048)
        ref = _run(_make(model, 2, 512, sc), sc, mic, g, 512, 2)
        worst = 0.0
        for b in (64, 192, 2048):
            sc2 = Sidechain()
            sc2.prepare(2048)
            y = _run(_make(model, 2, b, sc2), sc2, mic, g, b, 2)
            n = min(y.shape[-1], ref.shape[-1])
            worst = max(worst, float(np.max(np.abs(y[:, :n] - ref[:, :n]))) /
                        float(np.max(np.abs(ref))))
        check(f"{model}: same output at 64 / 192 / 512 / 2048", worst < 1e-9,
              f"worst relative error {worst:.1e}")


def test_talkbox_speaker_loop(check):
    print("\n[TB4] talkbox: speakers feeding the mic cannot run away")
    voice = _voice("aiuaiu", seg=0.5, gap=0.1)
    g = _guitar(len(voice), 0.3)
    block = 256
    for model in FX_TYPES["talkbox"].MODELS:
        for loop_gain in (0.5, 2.0):
            sc = Sidechain()
            sc.prepare(block)
            fx = _make(model, sc=sc)
            n = (len(g) // block) * block
            prev = np.zeros(block)
            out = np.zeros(n)
            for i in range(0, n, block):
                # The mic hears you plus the speaker playing the last block.
                sc.write(voice[i:i + block] + loop_gain * prev, block)
                y = fx.process(g[None, i:i + block].copy())[0]
                out[i:i + block] = y
                prev = y
            ok = np.all(np.isfinite(out))
            pk = float(np.max(np.abs(out))) / float(np.max(np.abs(g)))
            check(f"{model}, mic hears speakers at {loop_gain:g}x: finite, peak under 4x the guitar",
                  ok and pk < 4.0, f"peak {pk:.2f}x")


def test_talkbox_engine(check):
    print("\n[TB5] talkbox in the engine: mic sidechain and voice fade")
    from amprack.engine import Engine
    n = 256
    e = Engine(tempfile.mkdtemp(prefix="tanin-tb-"))
    e.samplerate = SR
    e.active_blocksize = n
    # Anas's wiring: mic on Input 1, guitar on Input 2.
    e.guitar.input_channel = 1
    e.vocals.input_channel = 0
    e.chain["loop"].set_type(0, "talkbox")
    e._prepare_buffers(2)
    voice = _voice("aiu", seg=0.6, gap=0.2)
    g = _guitar(len(voice), 0.3)
    ind = np.zeros((n, 2), dtype=np.float32)
    od = np.zeros((n, 2), dtype=np.float32)
    claimed = []
    gains = []
    for i in range(0, len(g) - n + 1, n):
        ind[:, 0] = voice[i:i + n]
        ind[:, 1] = g[i:i + n]
        e._callback(ind, od, n, None, None)
        claimed.append(e.sidechain.claimed)
        gains.append(e._voice_gain)
    check("engine copies the Vocals input into the mic sidechain every block",
          np.allclose(e.sidechain.buf[:n], ind[:, 0]), "")
    check("a talkbox in the loop rack claims the mic", all(claimed), "")
    mid = int(0.4 * SR / n)
    check("dry voice fades out while the mouth is open (Voice = 0)",
          min(gains[mid:mid + 10]) < 0.05, f"voice gain {min(gains[mid:mid + 10]):.3f}")
    check("no callback errors", e.callback_errors == 0, e.last_callback_exc)
    # Without a talkbox the voice is untouched.
    e.chain["loop"].set_type(0, "")
    for _ in range(20):
        e._callback(ind, od, n, None, None)
    check("no talkbox: voice gain returns to 1", abs(e._voice_gain - 1.0) < 1e-6,
          f"{e._voice_gain:.4f}")
    e.sidechain.clear()


def test_talkbox_cpu(check):
    print("\n[TB6] talkbox: CPU")
    mic = _voice("aiu", seg=1.0, gap=0.0)
    g = _guitar(len(mic))
    for model in FX_TYPES["talkbox"].MODELS:
        for block in (128, 256):
            sc = Sidechain()
            sc.prepare(block)
            fx = _make(model, 2, block, sc)
            _run(fx, sc, mic[:block * 20], g[:block * 20], block, 2)   # JIT warm-up
            t0 = time.perf_counter()
            _run(fx, sc, mic, g, block, 2)
            load = (time.perf_counter() - t0) / (len(g) / SR)
            check(f"{model} (stereo) at block {block} under 15% of real time", load < 0.15,
                  f"{load * 100:.1f}%")


def run_all(check):
    for t in (test_talkbox_follows_mouth, test_talkbox_gate, test_talkbox_invariance,
              test_talkbox_speaker_loop, test_talkbox_engine, test_talkbox_cpu):
        try:
            t(check)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(t.__name__, False, f"raised {type(e).__name__}: {e}")
    sidechain.MIC.clear()
