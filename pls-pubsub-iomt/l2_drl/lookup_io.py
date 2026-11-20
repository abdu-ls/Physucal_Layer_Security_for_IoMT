"""Locate and load the L1 SOP lookup table."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import RESULTS_DIR
from l2_drl.env import Lookup


def find_lookup(path=None) -> Path:
    if path:
        return Path(path)
    base = RESULTS_DIR / "l1"
    cands = sorted(base.glob("l1_*/sop_lookup.csv")) if base.exists() else []
    if not cands:
        raise FileNotFoundError("No sop_lookup.csv found. Run `python -m l1_phy.run_l1 --part lookup` first.")
    return cands[-1]


def load_lookup(path=None):
    p = find_lookup(path)
    return Lookup(pd.read_csv(p)), p
