#!/usr/bin/env python
"""Read one eval_label.py JSON object on stdin, print its loss to 4 decimals, or nothing.

A separate file rather than a python -c inside the driver: the inline version needed nested quotes inside a
single-quoted shell string and broke on them, which silently emptied the base_val_loss column.

Prints the empty string when the label had no validation gaps (n_gaps 0) or the JSON is unusable, so the
caller records "missing" rather than a misleading 0.0000.
"""
import json
import sys

try:
    d = json.load(sys.stdin)
    print(f"{d['loss']:.4f}" if d.get("n_gaps") else "")
except Exception:
    print("")
