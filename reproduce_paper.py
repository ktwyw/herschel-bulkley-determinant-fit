#!/usr/bin/env python3
"""Reproduce Figures 1-5 of Mullineux (2008) plus a synthetic demonstration."""

import argparse
import json
from pathlib import Path

from hb_mullineux.figures import reproduce_all

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs"))
    p.add_argument("--dpi", type=int, default=160)
    a = p.parse_args()
    print(json.dumps(reproduce_all(a.output, dpi=a.dpi), indent=2))
    print(f"\nFigures written to {a.output.resolve()}")
