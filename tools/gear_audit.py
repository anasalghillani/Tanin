"""Audit song-tone gear against Tanin's mapping table.

Scans a folder for tone data (manifest ``.json`` files and ``.tone2014.xml``
files, as extracted from song packages), and reports:

  * every gear key seen, how often, and what Tanin maps it to (or "MISSING");
  * every knob name per gear key with the min and max value seen, and which
    Tanin control it lands on (or "unmatched").

The knob ranges are what ``KNOB_OVERRIDES`` in ``amprack/gearmap.py`` needs:
pedal knob ranges are not documented anywhere public, so run this on real
tone files and add an override wherever a range is not 0-100.

Usage:
    .venv\\Scripts\\python.exe tools\\gear_audit.py <folder> [--json out.json]
"""

from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amprack import gearmap  # noqa: E402
from amprack.dsp.fx import FX_TYPES  # noqa: E402

SLOTS = ("Amp", "Cabinet", *(f"PrePedal{i}" for i in range(1, 5)),
         *(f"PostPedal{i}" for i in range(1, 5)), *(f"Rack{i}" for i in range(1, 5)))


def _gear_lists_json(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("GearList", "gearList") and isinstance(v, dict):
                yield v
            else:
                yield from _gear_lists_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _gear_lists_json(v)


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _gear_lists_xml(path):
    root = ET.parse(path).getroot()
    for gl in root.iter():
        if _local(gl.tag) != "GearList":
            continue
        gear = {}
        for slot in gl:
            name = _local(slot.tag)
            key, knobs = None, {}
            for ch in slot:
                t = _local(ch.tag)
                if t in ("PedalKey", "Key") and ch.text:
                    key = ch.text.strip()
                elif t == "KnobValues":
                    for kv in ch:
                        kk = vv = None
                        for e in kv:
                            if _local(e.tag) == "Key":
                                kk = (e.text or "").strip()
                            elif _local(e.tag) == "Value":
                                vv = e.text
                        if kk and vv is not None:
                            try:
                                knobs[kk] = float(vv)
                            except ValueError:
                                pass
            if key:
                gear[name] = {"Key": key, "Knobs": knobs}
        yield gear


def scan(folder):
    for dirpath, _, files in os.walk(folder):
        for fn in files:
            p = os.path.join(dirpath, fn)
            low = fn.lower()
            try:
                if low.endswith(".json"):
                    with open(p, encoding="utf-8", errors="replace") as f:
                        yield from _gear_lists_json(json.load(f))
                elif low.endswith(".xml") and "tone" in low:
                    yield from _gear_lists_xml(p)
            except (OSError, ValueError, ET.ParseError) as e:
                print(f"  skip {p}: {e}", file=sys.stderr)


def audit(folder):
    seen = defaultdict(int)
    knobs = defaultdict(lambda: defaultdict(lambda: [float("inf"), float("-inf"), 0]))
    tones = 0
    for gear in scan(folder):
        tones += 1
        for slot in SLOTS:
            s = gear.get(slot)
            if not isinstance(s, dict):
                continue
            key = s.get("Key") or s.get("PedalKey")
            if not key:
                continue
            seen[key] += 1
            for kn, v in gearmap._knobs(s).items():
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                r = knobs[key][kn]
                r[0], r[1], r[2] = min(r[0], v), max(r[1], v), r[2] + 1

    report = {"tones": tones, "gear": {}}
    for key in sorted(seen):
        if key.startswith(("Amp_", "DI_Amp")):
            m, exact = gearmap.lookup_amp(key)
            target, cls = f"amp {m}", None
        elif "Cab" in key:
            cm, mic, pos, exact = gearmap.lookup_cab(key)
            target, cls = f"cab {cm} / {mic}", None
        else:
            hit = gearmap.lookup_fx(key)
            if hit is None:
                target, exact, cls = "MISSING", False, None
            else:
                t, m, _, exact = hit
                target, cls = f"{t} {m}", FX_TYPES[t]
        kinfo = {}
        idx = gearmap._param_index(cls) if cls else {}
        for kn, (lo, hi, n) in sorted(knobs[key].items()):
            if cls is not None:
                p = idx.get(gearmap.knob_suffix(kn))
                if p is None and cls.TYPE == "geq":
                    p = gearmap._geq_band(cls, hit[1], kn)
                dest = p.name if p is not None else "unmatched"
            elif key.startswith(("Amp_", "DI_Amp")):
                dest = gearmap.AMP_KNOBS.get(gearmap.knob_suffix(kn), "unmatched")
            else:
                dest = "-"
            kinfo[kn] = {"min": lo, "max": hi, "count": n, "to": dest}
        report["gear"][key] = {"count": seen[key], "to": target,
                               "exact": bool(exact), "knobs": kinfo}
    return report


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    rep = audit(argv[0])
    if "--json" in argv:
        with open(argv[argv.index("--json") + 1], "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
    print(f"{rep['tones']} tones scanned, {len(rep['gear'])} gear keys")
    for key, g in rep["gear"].items():
        flag = "" if g["exact"] else ("   <-- MISSING" if g["to"] == "MISSING"
                                      else "   <-- guessed")
        print(f"{key:40s} x{g['count']:<4d} -> {g['to']}{flag}")
        for kn, k in g["knobs"].items():
            odd = "" if 0.0 <= k["min"] and k["max"] <= 100.0 else "   (outside 0-100)"
            print(f"    {kn:44s} {k['min']:8.2f} .. {k['max']:8.2f} -> {k['to']}{odd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
