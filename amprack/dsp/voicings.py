"""Circuit-modelled amp voicings and extra cabinets.

The four original amps (Clean, Crunch, Lead, Metal) are generic. These are
specific *families* of amp, each described by the facts that make it sound
the way it does: how many gain stages and how hard they are pushed, the
coupling filters between them, which passive tone stack it has and with
which parts (see tonestack.py), whether that stack sits before or after the
gain stages, whether it has a bright cap, whether it has a master volume
(and so breaks up in the power amp when turned up), and whether the power
amp is push-pull or single-ended / class A.

Names are Tanin's own and describe the circuit family, not a brand.
"""

from __future__ import annotations

from .blocks import AMP_MODELS, CAB_MODELS, MIC_TILT, AmpModel

_V = "Vintage"
_C = "Clean"
_H = "High gain"
_B = "Bass"

NEW_AMPS = {
    # -- American, tweed era ----------------------------------------------
    "Tweed Lux": AmpModel(
        "Tweed Lux", stages=2, stage_gain=0.8, bias=0.16, input_hp=40.0,
        coupling_lp=7500.0, tone_bass=100.0, tone_mid=500.0, tone_mid_q=0.6,
        tone_treble=3000.0, out_lp=6500.0, gain_range=(1.2, 45.0), stage_hp=25.0,
        single_tone=True, power=1.5, power_asym=0.12, master=False, family=_V),
    "Tweed Bass": AmpModel(
        "Tweed Bass", stages=3, stage_gain=0.8, bias=0.12, input_hp=45.0,
        coupling_lp=8500.0, tone_bass=100.0, tone_mid=500.0, tone_mid_q=0.6,
        tone_treble=3000.0, out_lp=7000.0, gain_range=(1.2, 70.0), stage_hp=25.0,
        stack="tweed", power=1.1, bright_db=3.0, master=False, family=_V),
    # -- American, blackface: big clean headroom, scooped -----------------
    "Blackface": AmpModel(
        "Blackface", stages=2, stage_gain=0.6, bias=0.05, input_hp=50.0,
        coupling_lp=12000.0, tone_bass=100.0, tone_mid=500.0, tone_mid_q=0.6,
        tone_treble=3000.0, out_lp=9000.0, gain_range=(1.0, 30.0), stage_hp=20.0,
        stack="blackface", power=0.55, bright_db=4.0, master=False, family=_C),
    # -- British -----------------------------------------------------------
    "Brit 45": AmpModel(
        "Brit 45", stages=3, stage_gain=0.8, bias=0.12, input_hp=55.0,
        coupling_lp=8500.0, tone_bass=110.0, tone_mid=650.0, tone_mid_q=0.7,
        tone_treble=2800.0, out_lp=6800.0, gain_range=(1.2, 80.0), stage_hp=30.0,
        stack="marshall", power=1.1, bright_db=2.0, master=False, family=_V),
    "Brit Plexi": AmpModel(
        "Brit Plexi", stages=3, stage_gain=0.85, bias=0.12, input_hp=70.0,
        coupling_lp=8000.0, tone_bass=110.0, tone_mid=650.0, tone_mid_q=0.7,
        tone_treble=2800.0, out_lp=6600.0, gain_range=(1.5, 130.0), stage_hp=40.0,
        stack="marshall", power=1.3, bright_db=4.0, master=False, family=_V),
    "Brit 800": AmpModel(
        "Brit 800", stages=4, stage_gain=0.95, bias=0.12, input_hp=85.0,
        coupling_lp=7200.0, tone_bass=120.0, tone_mid=700.0, tone_mid_q=0.8,
        tone_treble=2700.0, out_lp=6200.0, gain_range=(3.0, 320.0), stage_hp=55.0,
        stack="marshall", power=0.7, bright_db=2.0, family=_H),
    "Brit Modern": AmpModel(
        "Brit Modern", stages=4, stage_gain=1.0, bias=0.1, input_hp=100.0,
        coupling_lp=6800.0, tone_bass=120.0, tone_mid=750.0, tone_mid_q=0.8,
        tone_treble=2800.0, out_lp=6000.0, gain_range=(5.0, 700.0), stage_hp=75.0,
        stack="marshall", power=0.6, family=_H),
    "Chime 30": AmpModel(
        "Chime 30", stages=2, stage_gain=0.8, bias=0.15, input_hp=60.0,
        coupling_lp=10000.0, tone_bass=110.0, tone_mid=650.0, tone_mid_q=0.7,
        tone_treble=3200.0, out_lp=8000.0, gain_range=(1.2, 60.0), stage_hp=30.0,
        stack="vox", power=1.4, power_asym=0.15, bright_db=3.0, master=False,
        family=_V),
    "Citrus": AmpModel(
        "Citrus", stages=4, stage_gain=0.95, bias=0.12, input_hp=70.0,
        coupling_lp=6500.0, tone_bass=120.0, tone_mid=600.0, tone_mid_q=0.8,
        tone_treble=2600.0, out_lp=5800.0, gain_range=(2.5, 400.0), stage_hp=40.0,
        stack="orange", power=0.9, family=_H),
    "Hi-Power": AmpModel(
        "Hi-Power", stages=2, stage_gain=0.65, bias=0.06, input_hp=55.0,
        coupling_lp=11000.0, tone_bass=100.0, tone_mid=600.0, tone_mid_q=0.7,
        tone_treble=3000.0, out_lp=8500.0, gain_range=(0.8, 35.0), stage_hp=25.0,
        stack="hiwatt", power=0.5, family=_C),
    "Boutique": AmpModel(
        "Boutique", stages=3, stage_gain=0.8, bias=0.14, input_hp=55.0,
        coupling_lp=9000.0, tone_bass=110.0, tone_mid=600.0, tone_mid_q=0.7,
        tone_treble=3000.0, out_lp=7500.0, gain_range=(1.2, 100.0), stage_hp=30.0,
        stack="tweed", power=1.3, power_asym=0.1, master=False, family=_V),
    # -- California --------------------------------------------------------
    "Cali Clean": AmpModel(
        "Cali Clean", stages=2, stage_gain=0.65, bias=0.06, input_hp=45.0,
        coupling_lp=11000.0, tone_bass=100.0, tone_mid=500.0, tone_mid_q=0.6,
        tone_treble=3000.0, out_lp=8500.0, gain_range=(0.8, 25.0), stage_hp=20.0,
        stack="mesa", power=0.45, bright_db=3.0, family=_C),
    "Cali Lead": AmpModel(
        "Cali Lead", stages=5, stage_gain=0.95, bias=0.1, input_hp=80.0,
        coupling_lp=6800.0, tone_bass=110.0, tone_mid=700.0, tone_mid_q=0.8,
        tone_treble=2800.0, out_lp=6000.0, gain_range=(4.0, 900.0), stage_hp=60.0,
        stack="mesa", stack_pre=True,
        geq=((80.0, 3.0), (240.0, -2.0), (750.0, -4.0), (2200.0, 1.5), (6600.0, 3.0)),
        power=0.5, family=_H),
    "Cali Rectified": AmpModel(
        "Cali Rectified", stages=4, stage_gain=1.05, bias=0.08, input_hp=70.0,
        coupling_lp=6000.0, tone_bass=130.0, tone_mid=800.0, tone_mid_q=1.0,
        tone_treble=3000.0, out_lp=5600.0, gain_range=(6.0, 900.0), stage_hp=60.0,
        stack="modern", power=1.0, family=_H),
    # -- Modern high gain --------------------------------------------------
    "Hot Rod": AmpModel(
        "Hot Rod", stages=5, stage_gain=1.0, bias=0.09, input_hp=140.0,
        coupling_lp=6400.0, tone_bass=130.0, tone_mid=800.0, tone_mid_q=1.0,
        tone_treble=3000.0, out_lp=5800.0, gain_range=(8.0, 1200.0), stage_hp=100.0,
        stack="modern", power=0.6, family=_H),
    "Teutonic": AmpModel(
        "Teutonic", stages=5, stage_gain=1.0, bias=0.08, input_hp=120.0,
        coupling_lp=6600.0, tone_bass=130.0, tone_mid=750.0, tone_mid_q=1.0,
        tone_treble=3000.0, out_lp=6000.0, gain_range=(8.0, 1000.0), stage_hp=90.0,
        stack="modern", geq=((700.0, 2.5), (3500.0, -1.5)), power=0.5, family=_H),
    # -- Solid state and direct --------------------------------------------
    "Jazz SS": AmpModel(
        "Jazz SS", stages=1, stage_gain=0.5, bias=0.0, input_hp=40.0,
        coupling_lp=16000.0, tone_bass=120.0, tone_mid=700.0, tone_mid_q=0.6,
        tone_treble=4000.0, out_lp=12000.0, gain_range=(0.4, 6.0), shaper="tanh",
        stage_hp=15.0, family=_C),
    "Studio DI": AmpModel(
        "Studio DI", stages=1, stage_gain=0.4, bias=0.0, input_hp=20.0,
        coupling_lp=20000.0, tone_bass=100.0, tone_mid=1000.0, tone_mid_q=0.5,
        tone_treble=6000.0, out_lp=19000.0, gain_range=(0.8, 5.0), shaper="tanh",
        stage_hp=10.0, family=_C),
    "Tube Pre": AmpModel(
        "Tube Pre", stages=1, stage_gain=0.6, bias=0.1, input_hp=25.0,
        coupling_lp=16000.0, tone_bass=100.0, tone_mid=800.0, tone_mid_q=0.5,
        tone_treble=5000.0, out_lp=15000.0, gain_range=(0.6, 12.0), stage_hp=15.0,
        family=_C),
    "Bass Tube": AmpModel(
        "Bass Tube", stages=2, stage_gain=0.7, bias=0.1, input_hp=25.0,
        coupling_lp=9000.0, tone_bass=60.0, tone_mid=500.0, tone_mid_q=0.7,
        tone_treble=3000.0, out_lp=7000.0, gain_range=(0.8, 35.0), stage_hp=15.0,
        power=0.8, family=_B),
    "Bass SS": AmpModel(
        "Bass SS", stages=1, stage_gain=0.5, bias=0.0, input_hp=25.0,
        coupling_lp=14000.0, tone_bass=50.0, tone_mid=600.0, tone_mid_q=0.7,
        tone_treble=4000.0, out_lp=10000.0, gain_range=(0.6, 12.0), shaper="tanh",
        stage_hp=12.0, family=_B),
    "DI Drive": AmpModel(
        "DI Drive", stages=2, stage_gain=0.8, bias=0.04, input_hp=30.0,
        coupling_lp=7000.0, tone_bass=70.0, tone_mid=700.0, tone_mid_q=0.8,
        tone_treble=3000.0, out_lp=4500.0, gain_range=(1.0, 90.0), shaper="tanh",
        stage_hp=20.0, family=_B),
}

NEW_CABS = {
    # (hp_hz, hp_q, lp_hz, lp_q, [(peak_hz, q, db), ...])  -- at most 5 peaks
    "1x10 Tweed": (110.0, 0.7, 5200.0, 0.8,
                   [(220.0, 1.0, 2.0), (1100.0, 1.2, 3.0), (2600.0, 1.6, 4.0),
                    (4200.0, 2.0, -5.0)]),
    "1x12 Open": (90.0, 0.7, 5000.0, 0.8,
                  [(160.0, 1.0, 2.0), (700.0, 1.2, -3.0), (2400.0, 1.5, 5.0),
                   (3900.0, 2.0, -6.0)]),
    "2x12 Open": (80.0, 0.7, 5400.0, 0.8,
                  [(130.0, 1.0, 2.5), (600.0, 1.3, -2.0), (2800.0, 1.6, 5.5),
                   (4500.0, 2.2, -5.0)]),
    "4x10 Tweed": (75.0, 0.8, 4800.0, 0.8,
                   [(120.0, 1.0, 3.0), (450.0, 1.2, -2.0), (1800.0, 1.4, 3.0),
                    (3500.0, 2.0, -6.0)]),
    "4x12 Greenback": (80.0, 0.9, 4200.0, 0.9,
                       [(110.0, 1.0, 3.5), (500.0, 1.2, -1.0), (1500.0, 1.2, 3.5),
                        (2500.0, 1.6, 2.0), (3600.0, 2.2, -7.0)]),
    "4x12 V30": (75.0, 1.0, 5000.0, 1.0,
                 [(100.0, 1.0, 4.5), (400.0, 1.0, -3.0), (1100.0, 1.6, -3.0),
                  (2600.0, 1.5, 6.0), (4300.0, 2.5, -8.0)]),
    "2x12 Closed": (80.0, 1.0, 4800.0, 1.0,
                    [(120.0, 1.1, 4.0), (500.0, 1.2, -2.5), (2200.0, 1.5, 4.0),
                     (4000.0, 2.2, -7.0)]),
    "1x12 Closed": (95.0, 1.0, 4800.0, 0.9,
                    [(150.0, 1.2, 3.5), (800.0, 1.3, -3.0), (2300.0, 1.5, 4.0),
                     (3800.0, 2.0, -6.0)]),
    "2x15 Jazz": (60.0, 0.8, 6500.0, 0.8,
                  [(90.0, 1.0, 2.0), (1000.0, 1.0, -2.0), (3000.0, 1.5, 3.0)]),
    "8x10 Bass": (45.0, 0.9, 3800.0, 0.9,
                  [(80.0, 1.0, 3.0), (250.0, 1.0, -2.0), (800.0, 1.2, 2.0),
                   (2000.0, 1.5, -4.0)]),
    "4x10 Bass": (50.0, 0.9, 7000.0, 0.7,
                  [(90.0, 1.0, 2.5), (400.0, 1.1, -2.0), (3000.0, 1.5, 2.0),
                   (5500.0, 2.0, 3.0)]),
    "1x15 Bass": (40.0, 0.8, 3500.0, 0.8,
                  [(70.0, 1.0, 4.0), (300.0, 1.0, -1.5), (900.0, 1.2, 1.5)]),
    # Not guitar cabs at all: the speakers some song tones play through.
    "Boombox": (180.0, 0.8, 6000.0, 0.9,
                [(350.0, 1.0, 4.0), (1500.0, 1.0, -2.0), (4000.0, 1.5, 3.0)]),
    "Gramophone": (600.0, 0.7, 3000.0, 1.2,
                   [(900.0, 1.5, 6.0), (1800.0, 2.0, 4.0)]),
    "Console Radio": (250.0, 0.8, 3500.0, 1.0,
                      [(500.0, 1.2, 3.0), (1800.0, 1.4, 2.0)]),
    "Jukebox": (70.0, 0.8, 6000.0, 0.8,
                [(120.0, 1.0, 4.0), (2500.0, 1.0, 2.0)]),
    "PA Speaker": (65.0, 0.9, 12000.0, 0.7,
                   [(100.0, 1.0, 1.5), (2500.0, 1.5, 2.0)]),
    "Hi-Fi": (35.0, 0.7, 16000.0, 0.7, [(80.0, 1.0, 1.0)]),
}

NEW_MICS = {
    "Tube": (1.5, 2.0),   # large-diaphragm valve condenser: full and airy
}

AMP_MODELS.update(NEW_AMPS)
CAB_MODELS.update(NEW_CABS)
MIC_TILT.update(NEW_MICS)
