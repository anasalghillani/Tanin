"""Song-tone gear mapping: turn a game tone's gear list into a Tanin preset.

A song tone names its gear by key (``Amp_...``, ``Cab_...``, ``Pedal_...``,
``Rack_...``) in slots Amp, Cabinet, PrePedal1-4, PostPedal1-4 and Rack1-4,
each with a dict of knob values (``Knobs`` or ``KnobValues``) whose names end
in the control's name, e.g. ``..._Gain``.

This table lives in Tanin (not in the Play-Along app) so the harness can check
that every target it names really exists, and so any app that uses Tanin gets
the same translation.

What is known and what is not:
  * Gear keys come from public tone-file references and are reliable.
  * Amp knobs are 0-100 and are well documented.
  * Pedal knob names and ranges are NOT documented anywhere public. Knobs are
    matched by name against each Tanin parameter's name and aliases, and
    values are read as 0-100 unless they fall outside it. ``tools/gear_audit.py``
    reports every knob name and value range seen in real tone files, and
    ``KNOB_OVERRIDES`` below is where a confirmed range goes.

Nothing here touches audio. ``tone_to_preset`` returns a plain preset dict for
``Engine.apply_preset``, plus the list of anything it could not place.
"""

from __future__ import annotations

import math
import re

from .dsp.blocks import AMP_MODELS, CAB_MODELS, MIC_TILT
from .dsp.fx import FX_TYPES

# --------------------------------------------------------------------------
# Amps: game key -> Tanin voicing
# --------------------------------------------------------------------------

AMP_MAP: dict[str, str] = {
    # Fictional series keys
    "Amp_TW22": "Tweed Lux", "Amp_TW26": "Tweed Lux", "Amp_TW40": "Tweed Bass",
    "Amp_AT20": "Blackface", "Amp_AT120": "Jazz SS",
    "Amp_CS90": "Jazz SS", "Amp_CS100": "Jazz SS", "Amp_CS120": "Jazz SS",
    "Amp_BT15": "Brit 45", "Amp_BT30": "Brit Modern", "Amp_BT45": "Brit 45",
    "Amp_EN30": "Chime 30", "Amp_EN50": "Citrus",
    "Amp_CA38": "Cali Clean", "Amp_CA85": "Cali Lead", "Amp_CA100": "Cali Rectified",
    "Amp_GB38": "Teutonic", "Amp_GB50": "Teutonic", "Amp_GB100": "Teutonic",
    "Amp_HG100": "Hot Rod", "Amp_HG180": "Hot Rod", "Amp_HG500": "Cali Rectified",
    "DI_Amp_BassDriver": "DI Drive", "DI_Amp_TubePre": "Tube Pre",
    # Named keys: tweed era
    "Amp_Fender57CustomPro": "Tweed Lux", "Amp_Fender57CustomTwin": "Tweed Lux",
    "Amp_Fender59Bassman": "Tweed Bass", "Amp_FenderBassman": "Tweed Bass",
    "Amp_FenderChamp": "Tweed Lux", "Amp_FenderChampion600": "Tweed Lux",
    "Amp_MesaCaliforniaTweed": "Boutique",
    # Black-panel cleans
    "Amp_FenderDeluxeReverb": "Blackface", "Amp_FenderPrinceton": "Blackface",
    "Amp_FenderSuperReverb": "Blackface", "Amp_FenderTwinReverb": "Blackface",
    "Amp_FenderVibroverb": "Blackface", "Amp_FenderVibroking": "Blackface",
    "Amp_FenderProJunior": "Boutique", "Amp_FenderSupersonic": "Hot Rod",
    "Amp_FenderMH500": "Teutonic",
    # British
    "Amp_MarshallJTM45": "Brit 45", "Amp_Marshall1962Bluesbreaker": "Brit 45",
    "Amp_MarshallPlexi": "Brit Plexi", "Amp_MarshallMajor": "Brit Plexi",
    "Amp_MarshallJMP": "Brit Plexi",
    "Amp_MarshallJCM800": "Brit 800", "Amp_MarshallDSL100H": "Brit 800",
    "Amp_MarshallDSL15H": "Brit 800", "Amp_MarshallSilverJubilee": "Brit 800",
    "Amp_MarshallSlash": "Brit 800", "Amp_MarshallSlashGold": "Brit 800",
    "Amp_MarshallJCM900": "Brit Modern", "Amp_MarshallJVM": "Brit Modern",
    "Amp_MarshallJVM410H": "Brit Modern",
    "Amp_VoxAC30": "Chime 30", "Amp_VoxAC30Blue": "Chime 30",
    "Amp_Hiwatt": "Hi-Power", "Amp_Sunn1200": "Hi-Power",
    # Orange-style
    "Amp_OrangeAD30": "Citrus", "Amp_OrangeOR50": "Citrus", "Amp_OrangeOR120": "Citrus",
    "Amp_OrangeRockerverb": "Citrus", "Amp_OrangeThunderverb": "Citrus",
    "Amp_OrangeTinyTerror": "Citrus", "Amp_OrangeDualTerror": "Citrus",
    "Amp_OrangeAD200": "Bass Tube",
    # California
    "Amp_MesaCleanMKIII": "Cali Clean", "Amp_MesaTA30": "Cali Clean",
    "Amp_MesaBoogieMarkIIC": "Cali Lead", "Amp_MesaBoogieMarkIII": "Cali Lead",
    "Amp_MesaBoogieMarkIV": "Cali Lead", "Amp_MesaBoogieMarkV": "Cali Lead",
    "Amp_MesaLeadMKIII": "Cali Lead", "Amp_MesaModernTubeLead": "Cali Lead",
    "Amp_MesaDualRectifier": "Cali Rectified", "Amp_MesaTripleRectifier": "Cali Rectified",
    "Amp_MesaTripleCrown": "Cali Rectified",
    # High gain
    "Amp_Peavey5150": "Hot Rod", "Amp_Peavey5150II": "Hot Rod", "Amp_Soldano": "Hot Rod",
    "Amp_Randall": "Hot Rod", "Amp_JetCityJCA100H": "Hot Rod",
    "Amp_JetCityJCA20H": "Hot Rod", "Amp_PRSArchon": "Hot Rod", "Amp_CarvinV3M": "Hot Rod",
    "Amp_BognerEcstasy": "Teutonic", "Amp_DiezelVH4": "Teutonic",
    "Amp_EnglE650": "Teutonic", "Amp_EnglPowerball": "Teutonic",
    # Boutique / low-watt
    "Amp_Supro": "Boutique", "Amp_DrZMaz18": "Boutique", "Amp_DrZZWreck": "Boutique",
    "Amp_THDBiValve": "Boutique",
    # Clean solid state
    "Amp_RolandJC120": "Jazz SS",
    # Bass
    "Amp_Ampeg": "Bass Tube", "Amp_AmpegSVTVR": "Bass Tube", "Amp_AmpegB15": "Bass Tube",
    "Amp_FenderBassman300": "Bass Tube",
    "Amp_Acoustic360": "Bass SS", "Amp_Aguilar": "Bass SS",
}

# For keys not in the table: substrings, checked in order.
_AMP_HINTS = (
    ("bass", "Bass Tube"), ("di_", "Studio DI"), ("tweed", "Tweed Lux"),
    ("rect", "Cali Rectified"), ("mesa", "Cali Lead"), ("orange", "Citrus"),
    ("vox", "Chime 30"), ("ac30", "Chime 30"), ("marshall", "Brit 800"),
    ("plexi", "Brit Plexi"), ("fender", "Blackface"), ("jazz", "Jazz SS"),
    ("5150", "Hot Rod"), ("engl", "Teutonic"), ("diezel", "Teutonic"),
)

AMP_KNOBS = {
    "gain": "gain", "drive": "gain", "preamp": "gain", "sensitivity": "gain",
    "bass": "bass", "low": "bass",
    "mid": "mid", "middle": "mid", "mids": "mid",
    "treble": "treble", "high": "treble", "tone": "treble",
    "presence": "presence", "pres": "presence",
    "volume": "master", "master": "master", "loudness1": "master", "output": "master",
}

# --------------------------------------------------------------------------
# Cabinets: series + speaker layout -> Tanin cab, suffix -> mic
# --------------------------------------------------------------------------

CAB_SIZES = {  # (family hint, layout) -> cab; family None = any
    ("tweed", "112"): "1x12 Open", ("tweed", "110"): "1x10 Tweed",
    ("tweed", "410"): "4x10 Tweed", ("tweed", "212"): "2x12 Open",
    ("american", "112"): "1x12 Open", ("american", "212"): "2x12 Open",
    ("american", "410"): "4x10 Tweed", ("american", "115"): "2x15 Jazz",
    ("british", "412"): "4x12 Greenback", ("british", "212"): "2x12 Closed",
    ("british", "112"): "1x12 Closed",
    ("modern", "412"): "4x12 V30", ("modern", "212"): "2x12 Closed",
    ("modern", "112"): "1x12 Closed", ("modern", "215"): "2x15 Jazz",
    ("bass", "810"): "8x10 Bass", ("bass", "410"): "4x10 Bass",
    ("bass", "115"): "1x15 Bass", ("bass", "415"): "1x15 Bass",
    ("bass", "210"): "4x10 Bass", ("bass", "215"): "1x15 Bass",
    (None, "110"): "1x10 Tweed", (None, "112"): "1x12 Combo",
    (None, "212"): "2x12 Vintage", (None, "410"): "4x10 Tweed",
    (None, "412"): "4x12 Classic", (None, "115"): "2x15 Jazz",
    (None, "215"): "2x15 Jazz", (None, "810"): "8x10 Bass", (None, "18"): "1x8 Lo-Fi",
}

_CAB_FAMILY = (  # key prefix / substring -> family hint
    ("Bass_Cab", "bass"), ("Cab_TW", "tweed"), ("Cab_AT", "american"),
    ("Cab_CS", "american"), ("Cab_BT", "british"), ("Cab_Marshall", "british"),
    ("Cab_Vox", "british"), ("Cab_CA", "modern"), ("Cab_GB", "modern"),
    ("Cab_HG", "modern"), ("Cab_EN", "modern"), ("Cab_Mesa", "modern"),
    ("Cab_Orange", "modern"), ("Cab_Fender", "american"), ("Cab_Ampeg", "bass"),
)

CAB_SPECIAL = {  # novelty / non-guitar speakers by substring
    "boombox": "Boombox", "gramophone": "Gramophone", "radio": "Console Radio",
    "jukebox": "Jukebox", "pa": "PA Speaker", "hifi": "Hi-Fi", "lofi": "1x8 Lo-Fi",
}

MIC_SUFFIX = {"57": "Dynamic", "ribbon": "Ribbon", "condenser": "Condenser",
              "tube": "Tube", "dynamic": "Dynamic"}
MIC_POSITION = {"cone": 0.8, "edge": 0.3, "offaxis": 0.1}

# --------------------------------------------------------------------------
# Effects: game key -> (Tanin type, model, fixed params)
# --------------------------------------------------------------------------

_ENV = {"source": "Envelope"}
_LFO = {"source": "LFO"}

FX_MAP: dict[str, tuple[str, str, dict]] = {
    # Drives and fuzzes
    "Pedal_CustomDrive": ("dirt", "Asym", {}),
    "Pedal_Distortion": ("dirt", "Distortion", {}),
    "Pedal_MetalDistortion": ("dirt", "Zone", {}),
    "Pedal_ShredZone": ("dirt", "Zone", {}),
    "Pedal_BuzzOne": ("dirt", "Gated Fuzz", {}),
    "Pedal_BuzzToo": ("dirt", "Muff", {}),
    "Pedal_CaptFuzzle": ("dirt", "Muff", {}),
    "Pedal_FuzzWasHe": ("dirt", "Si Fuzz", {}),
    "Pedal_GermaniumDrive": ("dirt", "Ge Fuzz", {}),
    "Pedal_LineDrive": ("boost", "Clean", {}),
    "Pedal_RangeBooster": ("boost", "Treble", {}),
    "Rack_StudioBoost": ("boost", "Clean", {}),
    "Rack_StudioDist": ("dirt", "Distortion", {}),
    "Rack_StudioFuzz": ("dirt", "Muff", {}),
    "Rack_StudioOD": ("dirt", "Green", {}),
    # Wahs and filters (no expression pedal in a song tone: wahs ride the
    # pick attack unless the tone fixes a position)
    "Pedal_ModernWah": ("wah", "Modern", _ENV),
    "Pedal_UKWah": ("wah", "Vintage", _ENV),
    "Pedal_USWah": ("wah", "Classic", _ENV),
    "Rack_StudioWah": ("wah", "Classic", _ENV),
    "Pedal_AutoFilter": ("autofilter", "Bandpass", _ENV),
    "Pedal_BobFilter": ("autofilter", "Ladder", _ENV),
    "Rack_SynthFilter": ("autofilter", "Ladder", _LFO),
    "Pedal_LoFiFilter": ("lofi", "Radio", {}),
    "Pedal_BitCruncher": ("crusher", "Standard", {}),
    "Pedal_BitCrusher": ("crusher", "Standard", {}),
    # Pitch
    "Pedal_MultiPitch": ("pitch", "Shift", {}),
    "Rack_StudioPitch": ("pitch", "Harmony", {}),
    "Pedal_OctaveUp": ("octup", "Standard", {}),
    "Pedal_Octavius": ("octdown", "Analog", {}),
    # Modulation
    "Pedal_Chorus": ("chorus", "Classic", {}),
    "Pedal_Chorus20": ("chorus", "Ensemble", {}),
    "Rack_StudioChorus": ("chorus", "Tri", {}),
    "Rack_StudioModChorus": ("chorus", "Dimension", {}),
    "Pedal_OmniMod": ("chorus", "Tri", {}),
    "Pedal_80sFlanger": ("flanger", "Jet", {}),
    "Pedal_ClassicFlanger": ("flanger", "Classic", {}),
    "Pedal_ModernFlanger": ("flanger", "Studio", {}),
    "Pedal_VintageFlanger": ("flanger", "Vintage", {}),
    "Rack_StudioFlanger": ("flanger", "Studio", {}),
    "Rack_StudioModFlange": ("flanger", "Through-Zero", {}),
    "Pedal_Phaser": ("phaser", "Four-Stage", {}),
    "Pedal_ShaverPhaser": ("phaser", "Stone", {}),
    "Rack_StereoPhaser": ("phaser", "Twelve-Stage", {}),
    "Rack_StudioPhaser": ("phaser", "Six-Stage", {}),
    "Rack_StudioModPhase": ("phaser", "Six-Stage", {}),
    "Pedal_AmpTrem": ("tremolo", "Bias", {}),
    "Pedal_MultiTrem": ("tremolo", "Square", {}),
    "Pedal_Tremolo": ("tremolo", "Opto", {}),
    "Rack_StudioTremolo": ("tremolo", "Opto", {}),
    "Pedal_AmpVibe": ("vibe", "Chorus", {}),
    "Rack_StudioVibrato": ("vibe", "Vibrato", {}),
    "Pedal_BakedRotatoe": ("rotary", "Standard", {}),
    # Delays
    "Pedal_AnalogueDelay": ("echo", "Analog", {}),
    "Pedal_NPNDelay": ("echo", "Analog", {}),
    "Pedal_CosmicEcho": ("echo", "Multi-Head", {}),
    "Pedal_ModDelay": ("echo", "Memory", {}),
    "Pedal_NoFiEcho": ("echo", "Lo-Fi", {}),
    "Pedal_OilCanEcho": ("echo", "Oil Can", {}),
    "Pedal_ValveEcho": ("echo", "Tape", {}),
    "Rack_StudioDelay": ("echo", "Digital", {}),
    "Rack_StudioTapDelay": ("echo", "Multi-Tap", {}),
    "Rack_StudioTapeDelay": ("echo", "Tape", {}),
    "Rack_TapeEcho": ("echo", "Multi-Head", {}),
    # Reverbs
    "Pedal_DigitalVerb": ("verb", "Digital", {}),
    "Pedal_PlateVerb": ("verb", "Plate", {}),
    "Pedal_SpringReverb": ("verb", "Spring", {}),
    "Pedal_TubeSpring": ("verb", "Tank", {}),
    "Rack_StudioChamber": ("verb", "Chamber", {}),
    "Rack_StudioHallReverb": ("verb", "Hall", {}),
    "Rack_StudioPlate": ("verb", "Plate", {}),
    "Rack_StudioPlateReverb": ("verb", "Plate", {}),
    "Rack_StudioReverb": ("verb", "Hall", {}),
    "Rack_StudioRoomReverb": ("verb", "Room", {}),
    "Rack_StudioVerb": ("verb", "Digital", {}),
    # Dynamics and EQ
    "Pedal_Compression": ("comp", "Studio", {}),
    "Pedal_Limiter": ("comp", "Limiter", {}),
    "Pedal_Swole": ("comp", "Squash", {}),
    "Rack_StudioComp": ("comp", "Studio", {}),
    "Rack_StudioCompressor": ("comp", "Opto", {}),
    "Pedal_NoiseGate": ("gate", "Standard", {}),
    "Rack_StudioGate": ("gate", "Standard", {}),
    "Rack_StudioNoise": ("gate", "Standard", {}),
    "Pedal_EQ5": ("geq", "5-Band", {}),
    "Pedal_EQ8": ("geq", "8-Band", {}),
    "Rack_StudioEQ": ("geq", "7-Band", {}),
    "Rack_StudioGraphicEQ": ("geq", "10-Band", {}),
    "Rack_StudioParametricEQ": ("peq", "Standard", {}),
    # Acoustic
    "Pedal_AcousticEmulator": ("acoustic", "Dreadnought", {}),
    # Bass pedals
    "Bass_Pedal_MBComp": ("comp", "Opto", {}),
    "Bass_Pedal_BassEQ8": ("geq", "8-Band", {}),
    "Bass_Pedal_BassChorus": ("chorus", "Classic", {}),
    "Bass_Pedal_BassFuzz": ("dirt", "Muff", {}),
    "Bass_Pedal_BassOverdrive": ("dirt", "Blues", {}),
    "Bass_Pedal_BassDistortion": ("dirt", "Distortion", {}),
    "Bass_Pedal_BassSubOctave": ("octdown", "Deep", {}),
    "Bass_Pedal_EdenWTDI": ("boost", "Bass", {}),
    "Bass_Pedal_BassAutoFilter": ("autofilter", "Lowpass", _ENV),
    "Bass_Pedal_BassFlanger": ("flanger", "Classic", {}),
    "Bass_Pedal_BassEnbig": ("pitch", "Shift", {}),
}

# Keys not in the table: substrings, checked in order.
_FX_HINTS = (
    ("wah", ("wah", "Classic", _ENV)), ("autofilter", ("autofilter", "Bandpass", _ENV)),
    ("filter", ("autofilter", "Lowpass", _ENV)), ("octave", ("octdown", "Analog", {})),
    ("pitch", ("pitch", "Shift", {})), ("chorus", ("chorus", "Classic", {})),
    ("flang", ("flanger", "Classic", {})), ("phase", ("phaser", "Four-Stage", {})),
    ("trem", ("tremolo", "Opto", {})), ("vibe", ("vibe", "Chorus", {})),
    ("rotary", ("rotary", "Standard", {})), ("leslie", ("rotary", "Standard", {})),
    ("tape", ("echo", "Tape", {})), ("echo", ("echo", "Analog", {})),
    ("delay", ("echo", "Digital", {})), ("spring", ("verb", "Spring", {})),
    ("plate", ("verb", "Plate", {})), ("hall", ("verb", "Hall", {})),
    ("room", ("verb", "Room", {})), ("verb", ("verb", "Hall", {})),
    ("fuzz", ("dirt", "Muff", {})), ("dist", ("dirt", "Distortion", {})),
    ("drive", ("dirt", "Green", {})), ("boost", ("boost", "Clean", {})),
    ("comp", ("comp", "Studio", {})), ("limit", ("comp", "Limiter", {})),
    ("gate", ("gate", "Standard", {})), ("eq", ("geq", "7-Band", {})),
    ("crush", ("crusher", "Standard", {})), ("lofi", ("lofi", "Radio", {})),
    ("acoustic", ("acoustic", "Dreadnought", {})),
)

# Confirmed knob ranges go here once tools/gear_audit.py has seen real tone
# files: (gear key or "*", knob suffix lowercased) -> (param, "unit" | "raw").
# "unit" reads 0-100 as the parameter's full range; "raw" takes the value as
# the parameter's own unit (ms, Hz, dB).
KNOB_OVERRIDES: dict[tuple[str, str], tuple[str, str]] = {}

_HIGH_GAIN = {name for name, m in AMP_MODELS.items()
              if getattr(m, "family", "") == "High gain"} | {"Lead", "Metal"}

SLOT_GROUPS = (("pre", "PrePedal"), ("loop", "PostPedal"), ("rack", "Rack"))


# --------------------------------------------------------------------------
# Lookup helpers
# --------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def knob_suffix(name: str) -> str:
    """``Pedal_UKWah_Position`` -> ``position``."""
    return _norm(str(name).rsplit("_", 1)[-1])


def lookup_amp(key: str) -> tuple[str, bool]:
    """(Tanin amp model, exact match?)."""
    if key in AMP_MAP:
        return AMP_MAP[key], True
    low = str(key).lower()
    for hint, model in _AMP_HINTS:
        if hint in low:
            return model, False
    return "Crunch", False


def lookup_cab(key: str) -> tuple[str, str, float, bool]:
    """(cab model, mic, mic position, exact?) from a cabinet key."""
    key = str(key or "")
    low = key.lower()
    mic, pos = "Dynamic", 0.5
    for part in low.split("_"):
        if part in MIC_SUFFIX:
            mic = MIC_SUFFIX[part]
        if part in MIC_POSITION:
            pos = MIC_POSITION[part]
    for hint, cab in CAB_SPECIAL.items():
        if re.search(rf"(^|_){hint}", low):
            return cab, mic, pos, True
    fam = None
    for prefix, f in _CAB_FAMILY:
        if key.startswith(prefix) or prefix.lower() in low:
            fam = f
            break
    body = re.sub(r"^(bass_|ibex_)?cab_", "", key, flags=re.I).split("_")[0]
    m = re.search(r"(\d)x?(\d{1,2})", body)
    layout = m.group(1) + m.group(2) if m else ""
    if (fam, layout) in CAB_SIZES:
        return CAB_SIZES[(fam, layout)], mic, pos, True
    if (None, layout) in CAB_SIZES:
        return CAB_SIZES[(None, layout)], mic, pos, fam is None
    fallback = {"bass": "8x10 Bass", "tweed": "1x12 Open", "american": "2x12 Open",
                "british": "4x12 Greenback", "modern": "4x12 V30"}.get(fam, "4x12 Classic")
    return fallback, mic, pos, False


def lookup_fx(key: str) -> tuple[str, str, dict, bool] | None:
    """(Tanin effect type, model, fixed params, exact?) or None."""
    if key in FX_MAP:
        t, m, fixed = FX_MAP[key]
        return t, m, dict(fixed), True
    low = str(key).lower()
    # Bass/Ibex variants of a known pedal: match on the part after "Pedal_".
    tail = re.sub(r"^(bass_|ibex_)?(pedal|rack)_", "", low)
    for k, (t, m, fixed) in FX_MAP.items():
        if _norm(k.split("_", 1)[-1]) == _norm(tail):
            return t, m, dict(fixed), True
    for hint, (t, m, fixed) in _FX_HINTS:
        if hint in low:
            return t, m, dict(fixed), False
    return None


def _param_index(cls) -> dict[str, object]:
    idx = {}
    for p in cls.PARAMS:
        for nm in (p.name, p.label, *p.aliases):
            idx.setdefault(_norm(nm), p)
    return idx


def knob_value(p, v, how: str | None = None):
    """Translate one knob value onto Tanin parameter ``p``."""
    if p.kind == "choice":
        if isinstance(v, str):
            for c in p.choices:
                if _norm(c) == _norm(v):
                    return c
            return p.default
        fv = float(v)
        if fv.is_integer() and 0 <= fv < len(p.choices) and how != "unit":
            return p.choices[int(fv)]
        return p.from_unit(fv / 100.0)
    if p.kind == "bool":
        return float(v) >= 50.0 if float(v) > 1.0 else bool(v)
    fv = float(v)
    if how is None:
        # Signed dB controls (EQ bands, levels) are stored in dB; anything
        # outside 0-100 must be in the control's own unit; otherwise 0-100.
        if (p.unit == "dB" and p.lo < 0.0) or fv < 0.0 or fv > 100.0:
            how = "raw"
        else:
            how = "unit"
    if how == "raw":
        return p.clamp(fv)
    return p.clamp(p.from_unit(fv / 100.0))


def _geq_band(cls, model, kname):
    """Graphic EQ knobs are usually named by frequency (``..._400``,
    ``..._1.6k``, ``..._3200Hz``): pick the nearest Tanin band."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(k)?(?:hz)?$", str(kname).rsplit("_", 1)[-1].lower())
    if not m:
        return None
    f = float(m.group(1)) * (1000.0 if m.group(2) else 1.0)
    if f <= 0:
        return None
    bands = cls.bands(model)
    i = min(range(len(bands)), key=lambda j: abs(math.log(bands[j] / f)))
    return next(q for q in cls.PARAMS if q.name == f"b{i + 1}")


def _knobs(slot: dict) -> dict:
    k = slot.get("Knobs")
    if not isinstance(k, dict):
        k = slot.get("KnobValues")
    return k if isinstance(k, dict) else {}


# --------------------------------------------------------------------------
# Tone -> preset
# --------------------------------------------------------------------------

def gear_list(tone: dict) -> dict:
    """Accept a tone, a {"GearList": ...} wrapper or a bare gear list."""
    if not isinstance(tone, dict):
        return {}
    for k in ("GearList", "gearList", "Gear"):
        if isinstance(tone.get(k), dict):
            return tone[k]
    return tone


def tone_to_preset(tone: dict) -> tuple[dict, list[str]]:
    """Build a Tanin preset from a song tone. Returns (preset, notes); notes
    list anything unmapped or approximated, for the song screen."""
    gear = gear_list(tone)
    notes: list[str] = []
    preset: dict = {
        "gate": {"enabled": False}, "comp": {"enabled": False},
        "drive": {"enabled": False}, "ir": {"enabled": False},
        "mod": {"enabled": False}, "delay": {"enabled": False},
        "reverb": {"enabled": False}, "output": {"volume_db": -6.0},
    }

    # Amp
    amp = gear.get("Amp") if isinstance(gear.get("Amp"), dict) else {}
    akey = str(amp.get("Key") or amp.get("PedalKey") or "")
    if akey:
        model, exact = lookup_amp(akey)
        if not exact:
            notes.append(f"amp {akey}: no exact match, using {model}")
        a = {"enabled": True, "model": model}
        for kname, v in _knobs(amp).items():
            dst = AMP_KNOBS.get(knob_suffix(kname))
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if dst:
                a[dst] = max(0.0, min(1.0, fv / 100.0))
            else:
                notes.append(f"amp knob {kname}: not used")
        preset["amp"] = a
        if model in _HIGH_GAIN:
            preset["gate"] = {"enabled": True, "threshold_db": -58.0,
                              "attack_ms": 1.0, "release_ms": 90.0}
    else:
        preset["amp"] = {"enabled": False}

    # Cab
    cab = gear.get("Cabinet") if isinstance(gear.get("Cabinet"), dict) else {}
    ckey = str(cab.get("Key") or cab.get("PedalKey") or "")
    if ckey:
        cm, mic, pos, exact = lookup_cab(ckey)
        if not exact:
            notes.append(f"cab {ckey}: closest is {cm}")
        preset["cab"] = {"enabled": True, "model": cm, "mic": mic,
                         "position": pos, "distance": 0.3}
    else:
        preset["cab"] = {"enabled": bool(akey)}

    # Pedals and rack
    for rack, prefix in SLOT_GROUPS:
        slots = []
        for i in range(1, 5):
            s = gear.get(f"{prefix}{i}")
            if not isinstance(s, dict):
                continue
            key = str(s.get("Key") or s.get("PedalKey") or "")
            if not key:
                continue
            hit = lookup_fx(key)
            if hit is None:
                notes.append(f"{key}: no Tanin equivalent")
                continue
            t, m, params, exact = hit
            if not exact:
                notes.append(f"{key}: guessed as {FX_TYPES[t].LABEL} ({m})")
            cls = FX_TYPES[t]
            idx = _param_index(cls)
            params["model"] = m
            for kname, v in _knobs(s).items():
                suf = knob_suffix(kname)
                ov = KNOB_OVERRIDES.get((key, suf)) or KNOB_OVERRIDES.get(("*", suf))
                p = None
                how = None
                if ov:
                    p = next((q for q in cls.PARAMS if q.name == ov[0]), None)
                    how = ov[1]
                if p is None:
                    p = idx.get(suf)
                if p is None and t == "geq":
                    p = _geq_band(cls, m, kname)
                if p is None:
                    notes.append(f"{key} knob {kname}: no matching control")
                    continue
                try:
                    params[p.name] = knob_value(p, v, how)
                except (TypeError, ValueError):
                    notes.append(f"{key} knob {kname}: unreadable value {v!r}")
            slots.append({"type": t, "enabled": True, "params": params})
        while len(slots) < 4:
            slots.append({"type": "", "enabled": True, "params": {}})
        preset[rack] = {"enabled": True, "slots": slots[:4]}
    return preset, notes


def known_targets_ok() -> list[str]:
    """Every target named in the tables exists in Tanin. Returns problems."""
    bad = []
    for k, m in AMP_MAP.items():
        if m not in AMP_MODELS:
            bad.append(f"{k} -> amp {m}")
    for _, m in _AMP_HINTS:
        if m not in AMP_MODELS:
            bad.append(f"hint -> amp {m}")
    for m in (*CAB_SIZES.values(), *CAB_SPECIAL.values()):
        if m not in CAB_MODELS:
            bad.append(f"cab {m}")
    for m in MIC_SUFFIX.values():
        if m not in MIC_TILT:
            bad.append(f"mic {m}")
    for k, (t, m, fixed) in (*FX_MAP.items(), *_FX_HINTS):
        cls = FX_TYPES.get(t)
        if cls is None or m not in cls.MODELS:
            bad.append(f"{k} -> {t}/{m}")
            continue
        names = {p.name: p for p in cls.PARAMS}
        for pn, pv in fixed.items():
            if pn not in names or names[pn].clamp(pv) != pv:
                bad.append(f"{k} fixed {pn}={pv}")
    return bad
