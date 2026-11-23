# Tanin (طنين) — Handover

A real-time guitar amp and pedalboard for Windows, built in Python. It's a standalone project and has nothing to do with Rocksmith. It started in a session rooted in the Rocksmith folder only because that's where the `realtime-audio-hook` skill lived.

- **Location:** `%USERPROFILE%\Documents\AmpRack`
- **Run:** `Tanin.cmd`, or `.venv\Scripts\python.exe run.py`
- **List devices:** `.venv\Scripts\python.exe run.py --devices`
- **Tests:** `.venv\Scripts\python.exe tools\harness.py`. This is **100/100 passing** and must stay green.
- **GitHub:** https://github.com/anasalghillani/Tanin (public). Product name is **Tanin**;
  it was called AmpRack during development. The local folder is still `Documents\AmpRack`
  and the Python package is still `amprack` — renaming the folder breaks the venv's absolute
  paths, and renaming the package is churn with no user-visible benefit.
- **Fresh clone:** `py -3.12 -m venv .venv` then
  `.venv\Scripts\python.exe -m pip install -r requirements.txt`.

---

## User setup (verified on this machine)

| Thing | Value |
|---|---|
| Interface | PreSonus **AudioBox USB 96**, ASIO driver installed (`AudioBox ASIO Driver`) |
| Guitar | AudioBox **Input 1** (`ch0`) |
| Mic | AudioBox **Input 2** (`ch1`). Discord uses it for voice. |
| Discord voice | Input **and** output both on the AudioBox |
| Python | 3.12 at `...\Programs\Python\Python312`, run via `py -3.12`. Plain `python` is the Store stub. |
| Venv deps | numpy, scipy, sounddevice, soundfile, PySide6, numba |
| Driver panel | `C:\Program Files\PreSonus\Universal Control\Universal Control.exe` |

### User preferences
- **Don't use Voicemod.** The user explicitly rejected it, and rejected any approach that replaces their Discord mic with a virtual cable.
- They want to **talk on Discord and play at the same time**, with friends hearing Tanin through **screen share**. They share both the whole screen and the Tanin window.
- They want an easy, intuitive UI in the spirit of Guitar Rig: an original design, not a copy.

---

## Architecture

```
amprack/
  __init__.py       sets SD_ENABLE_ASIO=1 (must happen before any sounddevice import)
  dsp/core.py       RBJ biquads, OnePole, DCBlocker, SmoothedParam, DelayLine, LFO,
                    4x Oversampler, waveshapers, sanitize()
  dsp/blocks.py     NoiseGate, Compressor, Drive(5 models), Amp(Clean/Crunch/Lead/Metal),
                    Cabinet(5 voicings x 3 mics), IRLoader(overlap-save FFT),
                    Modulation(Chorus/Flanger/Phaser/Tremolo/Vibrato), Delay(4 modes),
                    Reverb(Freeverb: Room/Hall/Plate/Cathedral), Output(+safety limiter)
  dsp/chain.py      gate→comp→drive→amp→cab→ir (mono) → mod→delay→reverb→output (stereo)
                    PARAM_SPEC drives snapshot()/apply() for presets
  engine.py         Engine (stream + callback), Metronome, BroadcastOut (Stream Out),
                    device helpers: list_devices, is_asio, find_device,
                    broadcast_candidates, same_hardware, matching_input
  looper.py         one loop, overdub, single undo, ½ speed, reverse, decay
  recorder.py       ring buffer → background thread → wet stereo WAV + DI mono WAV
  tuner.py          InputRing + McLeod NSDF pitch detection
  presets.py        9 factory presets written to presets/*.json on first run
  ui/theme.py       palette + QSS, per-module accent colours
  ui/widgets.py     hand-painted Knob, PowerLED, LevelMeter (multi-channel, labelled),
                    LoopProgress, TunerDisplay, LoadIndicator
  ui/pedals.py      declarative KnobSpec/SelectSpec → PedalCard; IRCard subclass
  ui/main.py        MainWindow: header, Audio Setup panel, I/O strip, board, transport
  instrument.py     Instrument interface: prepare / process(mono, n) -> stereo /
                    snapshot / apply, enabled, input_channel (per instrument)
  instruments/      guitar.py (wraps dsp/chain.py), vocals.py (vocal chain +
                    5 vocal presets, apply_vocal_preset keeps key and range)
  mixer.py          Strip (volume, pan, mute, solo) + send matrix to 3 buses:
                    phones, record, stream
  dsp/autotune.py   McLeod detection every ~5 ms + real-time TD-PSOLA (numba),
                    fixed 12.5 ms latency, block-size invariant
  dsp/scales.py     keys, 12 scales (incl. Custom), C1..C6 range, snapping
  dsp/vocal_blocks.py  DeEsser, VocalEQ, Saturation, Doubler
  ui/vocals_panel.py   temporary Vocals tab in the dock (until phase 4)
  dsp/fx/sidechain.py  mic sidechain: engine copies the Vocals input into
                    sidechain.MIC each block, before any instrument runs
  dsp/fx/talkbox.py    Talkbox slot effect: Vowel (LPC formant tracking, 3
                    resonators) and Vocoder (16 bands). Level follows the
                    guitar, never the mic (speaker safe); auto noise-floor
                    gate; fades the dry voice to its Voice knob
tools/harness.py    offline test suite (suite_tests.py = phases 1-2,
                    talkbox_tests.py = talkbox + mic sidechain)
tools/bench_instruments.py  callback load with 3 instruments at 128/256
settings.json       saved devices (by NAME + API, not by index), channel, Stream Out
```

### Real-time rules (from the `realtime-audio-hook` skill; follow them)
- The callback never allocates anything beyond block size, takes no locks, does no I/O, and never raises. Failures turn into silence plus a counter.
- Knobs are plain attribute writes. Anything that resizes a buffer goes through `engine.post()`, a deque drained at the top of each block.
- GC is frozen and disabled while streaming.
- **DelayLine contract: read first, then write.** Both read methods require `delay >= n`. Breaking this makes every echo exactly one buffer late.
- Every effect must be bit-exact when bypassed and block-size invariant. The harness checks both.

---

## Instrument suite (phases 1-2, Nov 2026)

Plan: "Tanin Suite Plan" doc (6 phases). Phases 1 (engine foundation) and 2
(vocals + auto-tune) are in. Callback flow now: every enabled instrument reads
its own input channel and runs its chain -> mixer strips and sends -> phones /
record / stream buses -> looper (records phones, playback added to all three)
-> click -> tuner mute -> recorder (record bus) / Stream Out (stream bus) ->
output (phones bus). `engine.chain` and `engine.input_channel` still point at
the guitar, so the guitar UI is unchanged. Vocals default **off**, on In 2.

- numba compiles the PSOLA kernel in `AutoTune.prepare()` (control thread);
  first run after install takes a few seconds, then it is cached.
- CPU, measured on a Mac (callback with guitar fully loaded + vocals Hard Tune
  + a second full guitar chain standing in for piano): 256 samples about 41%
  mean / 52% p99; 128 samples about 63% mean / 80-100% p99. So three
  instruments want a 256 buffer; two (guitar + vocals) are fine at 128 (37%).
  Re-run `tools/bench_instruments.py` on the Windows PC for its own numbers.
- Tried replacing scipy `lfilter` with a numba kernel: only 4% saved, not
  bit-exact, reverted. The cost is spread over many small numpy calls.

## Hard-won facts. Don't re-learn these.

1. **ASIO only appears if `SD_ENABLE_ASIO` is set *before* sounddevice is imported.** It's now set in `amprack/__init__.py` and in `run.py`. If anything imports sounddevice first, ASIO vanishes silently and the app falls back to WASAPI at about 45 ms. In ad-hoc test scripts, `import amprack` comes first.
2. **PortAudio device indices are unstable.** The ASIO device showed up as index 18, then 23, then 18 again. Settings resolve devices with `find_device(name, api)`, and the stored index is only a fallback hint.
3. **The ASIO buffer size belongs to the driver, not the app.** It's currently set to **2048 samples, about 77 ms**. That's too slow to play through comfortably. The user needs to set 128 or 256 in Universal Control. The engine reads the driver's real buffer with `_probe_asio_blocksize()` and sizes all DSP to match. Measured with a small buffer, ASIO gave 11.4 ms round trip.
4. **WASAPI exclusive mode can't be negotiated on this interface** (it fails with "Invalid sample rate"). `Engine.start()` automatically falls back first to the device's own sample rate, then to shared mode, and tells the UI.
5. **The AudioBox driver is multi-client.** Verified while ASIO was running: every Windows output endpoint opens with zero gaps, and the AudioBox mic opens on WASAPI, WDM-KS, DirectSound and MME. This is why Discord voice and Tanin can share the box.
6. **ASIO never enters the Windows audio graph**, so Discord can't hear it. That's the reason Stream Out exists.
7. The channel selector used to need a stream restart to take effect. It now applies live, and the input meter shows every hardware channel.

---

## Stream Out (Discord): current design

`BroadcastOut` is a second, independent WASAPI or WDM-KS output stream that receives a copy of the final mix through a ring buffer.

- It runs on its **own clock**. A 0.5 s ring buffer sits at a 120 ms target fill, and drift is corrected by at most one sample per block, with a ±30 ms deadband. Simulated drift of ±1000 ppm over 10 minutes produced zero gaps (harness test 11).
- The first 0.5 s warmup is not counted as underruns.
- **Test** button: plays a 2 s, 660 Hz tone into the Stream Out device only.
- `broadcast_candidates()` **excludes the monitoring hardware**, which would cause a doubled guitar about 120 ms late. It ranks silent virtual sinks first (`_sink_score`). NVIDIA Broadcast is deliberately not treated as silent.
- Stream Out is **on by default** and the current device is `Speakers (Steam Streaming Microphone)`.
- Intended flow: Stream Out on → in Discord, share the **Tanin window** with sound → mic stays on the AudioBox.

### Status: NOT yet confirmed by the user
- The user hasn't yet reported whether friends can hear it after the last change. **Ask first.**
- **Whole-screen share is unverified.** Discord may capture only the default endpoint, which is the AudioBox. If it doesn't pick up a silent sink, the only options without extra software involve a tradeoff: send Stream Out to the AudioBox and hear a delayed second copy. Explain that tradeoff to the user before doing it.
- Possible follow-up: mix the AudioBox mic (`ch1`) into Stream Out as an option.

---

## Open items (priority order)

1. **Confirm Discord works** for both window share and whole-screen share (see above).
2. **User to lower the ASIO buffer** in Universal Control to 128 or 256. Then confirm the header latency turns green and the dropout counter stays at 0.
3. Verification ladder: the DSP is verified offline (rung 1), and the stream is verified live with no xruns (rung 3). **Rung 4 is still open:** the user playing a real guitar and judging the tone. Any feedback like "sounds off" should go into the harness as a numeric test before the DSP is touched.
4. Known DSP limit: at very large buffers (≥1024) the chorus centre delay is pushed past 14 ms by the read-before-write contract. This is physics, not a bug, and it goes away once the buffer is sane.

---

## Verification snapshot (2026-08-23)
- Harness: 73/73 (33 core checks plus the presets/IR, looper/recorder, meters/tuner-mute and DSP-audit tests). Bypass is bit-exact, block-size invariance has 0.0 error, delay timing is sample-accurate, aliasing is −96 dB, the tuner is within 1 cent, DSP load is 8–20%, and Stream Out drift is clean.
- Live on ASIO: 0 xruns and 0 callback errors. Looper record, overdub and undo work, and both WAVs write correctly.
- Two-channel capture confirmed that In 1 and In 2 carry different signals, so the hardware routing is fine.
