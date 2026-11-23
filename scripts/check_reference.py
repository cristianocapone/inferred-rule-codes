#!/usr/bin/env python3
"""Check regenerated numerical outputs against the original snapshot."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
failures = []
checked = 0
for source in sorted((ROOT / "reference/tables").glob("*.tex")):
    target = ROOT / "paper/supp_tables" / source.name
    if not target.exists() or target.read_bytes() != source.read_bytes():
        failures.append(str(target.relative_to(ROOT)))
    checked += 1
for source in sorted((ROOT / "reference/statistics").glob("*.json")):
    target = ROOT / "results" / source.name
    if not target.exists() or json.loads(target.read_text()) != json.loads(source.read_text()):
        failures.append(str(target.relative_to(ROOT)))
    checked += 1
if failures:
    sys.exit("Outputs differ from the frozen reference:\n" + "\n".join(failures))
print(f"All {checked} numerical outputs match the original snapshot.")
