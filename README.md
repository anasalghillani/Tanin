# Tanin — طنين

*Tanin (طنين): the ringing, sustained resonance of a struck string.*

A real-time guitar amp and pedalboard for Windows. Plug into your interface,
hear yourself through an amp with effects, loop over it, and record the result.

```bash
Tanin.cmd
```

Or `.venv\Scripts\python.exe run.py`. `run.py --devices` lists audio devices.

---

## First run: get the latency right

This is the only setup that matters. Everything else has a sensible default.

Open **Audio Setup** in the top-right and look at the latency readout in the
header. It is colour-coded:

| Reading | Feel |
|---|---|
| under 14 ms | green — feels like a real amp |
| 14–26 ms | amber — playable, slightly soft |
| over 26 ms | red — you will hear yourself late |

**Your AudioBox has an ASIO driver, and Tanin picks it automatically.** But
the ASIO buffer size belongs to the driver, not to this app — right now it is
set to **2048 samples (≈77 ms)**, which is far too slow to play through.

Fix it once:

1. Click **ASIO Control Panel…** in Audio Setup (opens PreSonus Universal Control).
2. Set the buffer size to **128** or **256** samples.
3. Back in Tanin, press **Start Audio**.

That should land you around 8–12 ms. If your machine starts clicking (the
header shows a dropout counter), go up one step.

Non-ASIO devices work too — pick one from the Input list and Tanin sets the
buffer itself — but expect roughly 45 ms on this interface, which is noticeably
laggy. ASIO is the path worth taking.

---

## The board

Signal flows left to right, top row then bottom row, exactly as drawn.

**Gate → Compressor → Drive → Amplifier → Cabinet → IR → Modulation → Delay → Reverb**

Click a module's LED to switch it in or out. Bypassed modules are provably
transparent — the test suite asserts bit-exactness, so switching one off cannot
colour your tone.

- **Amplifier** — Clean, Crunch, Lead, Metal. Cascaded gain stages with 4×
  oversampling, so the distortion stays harmonic instead of turning to
  aliased grit. *Sag* models power-supply droop: turn it up and loud notes
  compress and bloom the way a cranked valve amp does.
- **Cabinet** — five speaker voicings, three mic types, continuous mic
  position and distance. Load a real `.wav` impulse response in **IR Loader**
  if you'd rather.
- **Modulation** — Chorus, Flanger, Phaser, Tremolo, Vibrato.
- **Delay** — Digital, Analog, Tape (with wow), Ping-Pong.
- **Reverb** — Room, Hall, Plate, Cathedral.

Knobs: drag up/down, hold **Shift** for fine, **Ctrl** for coarse,
**double-click** to reset, scroll wheel to nudge.

## Looper

One loop, unlimited overdubs, one undo — the hardware-pedal workflow.

**Space** cycles: `REC → END REC → OVERDUB → PLAY → OVERDUB → …`

The loop records *after* the effects, so it captures the tone you played. Undo
removes the last overdub. *Decay* below 100% makes older layers fade as you
build. **½×** plays it back an octave down; **Rev** plays it backwards.

## Recorder

**RECORD** writes a stereo WAV of everything you hear. With *also save DI*
ticked it writes a second mono file of the raw guitar — same take, no
processing — so you can re-amp it later through a different rig.
Files land in `recordings/`.

## Streaming to Discord

**ASIO bypasses Windows audio completely.** It talks to the interface
directly, so Discord hears nothing from Tanin — not because anything is
misconfigured, but because the sound never enters the Windows audio graph.

**STREAM OUT**, at the right of the transport bar, gives Discord something to
capture. It is on by default.

### The setup that works

1. Leave **STREAM OUT** on. Its device defaults to a silent virtual sink
   (`Speakers (Steam Streaming Microphone)`) — you will not hear it.
2. Press **Test**. A 2-second tone plays into that device only. **You should
   not hear it.** If you do, pick a different device from the list.
3. In Discord, share the **Tanin window** and enable sound for it.
4. Leave Discord's Input and Output on the AudioBox as normal.

Your friends hear the guitar through the screen share, and your voice through
Discord's normal mic path, at the same time. Your AudioBox mic stays fully
available to Discord while Tanin is running — the driver is multi-client, so
ASIO and Windows can use the interface simultaneously.

### Choosing the device

Whatever you pick genuinely plays out of that device, so pick one you do not
listen to. The list is ordered with silent virtual sinks first and excludes
your monitoring interface outright — sending a second copy there would give
you the same guitar twice, about 120 ms apart.

### Sharing your whole screen

Discord's whole-screen audio capture may or may not include Tanin depending
on your Discord version. Sharing the Tanin window specifically is the
reliable route. If you need whole-screen sharing to carry the guitar, set
Stream Out to the device Discord is capturing — and accept that you will hear
the delayed second copy.

Stream Out runs on its own clock, so it uses a generous buffer. Listeners hear
it slightly later than you do, which costs you nothing. If the header shows
"stream out N gaps", that device cannot keep up; pick another.

## Tuner

Toggle with **T**. Green within ±3 cents. It reads the input directly, so it
works regardless of what the amp is doing.

## Keyboard

| Key | |
|---|---|
| `Space` | looper rec / play / overdub |
| `S` `U` `C` | loop stop / undo / clear |
| `R` | start or stop recording |
| `T` | tuner |
| `M` | metronome |
| `B` | global bypass |

---

## If something goes wrong

**Dropouts counter climbing** — raise the buffer size one step. The DSP itself
uses only 8–20% of the deadline, so dropouts almost always mean the buffer is
too small for the rest of the machine, not that the effects are too heavy.

**"Could not start"** — something else owns the interface. ASIO is exclusive by
nature, so close any DAW or game using it first. Tanin will fall back
automatically from exclusive to shared mode, and from a refused sample rate to
the device's own, and tells you when it does.

**Squealing / runaway feedback** — press **PANIC** to kill every delay and
reverb tail. There is also a brick-wall limiter on the output that cannot be
turned off; it exists so a feedback loop can't hurt your ears.

---

## Development

```bash
.venv\Scripts\python.exe tools\harness.py
```

33 offline checks covering bypass transparency, block-size invariance,
sample-accurate delay time, feedback stability, mid-stream parameter changes,
looper record/overdub/undo, tuner accuracy against synthesised tones,
distortion aliasing, the CPU budget, and stream-out clock drift. Run it before changing any DSP —
audio bugs do not raise exceptions, they just quietly sound wrong.

Layout:

```
amprack/
  dsp/core.py     biquads, delay lines, oversampling, smoothing
  dsp/blocks.py   the effects
  dsp/chain.py    signal routing
  engine.py       stream, callback, device handling
  looper.py       recording/overdub state machine
  recorder.py     ring buffer + background WAV writer
  tuner.py        NSDF pitch detection
  ui/             knobs, meters, pedal cards, main window
```

The audio callback allocates nothing that scales beyond one block, takes no
locks, and never raises — a failure degrades to silence plus a counter in the
status bar rather than a dead stream. Knob moves are plain attribute writes;
anything that would resize a buffer goes through a command queue drained at the
top of a block.
