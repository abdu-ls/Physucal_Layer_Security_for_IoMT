"""Shared helpers: config loading, unit conversion, path loss, result folders."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = Path(os.environ.get("PLS_RESULTS_DIR", ROOT / "results"))


def load_params(path: str | Path | None = None) -> dict:
    with open(path or CONFIG_DIR / "params.yaml") as f:
        return yaml.safe_load(f)


def load_topics(path: str | Path | None = None) -> dict:
    with open(path or CONFIG_DIR / "topics.yaml") as f:
        cfg = yaml.safe_load(f)
    for t in cfg["topics"]:
        t["epsilon"] = cfg["sensitivity_targets"][t["sensitivity"]]
        t["K"] = len(t["subscribers"])
    return cfg


def dbm_to_w(dbm: float) -> float:
    return 10 ** ((dbm - 30.0) / 10.0)


def pathloss_gain(d, phy: dict):
    """Linear power gain at distance d (m): free-space loss at 1 m, then exponent alpha."""
    lam = 3e8 / phy["carrier_hz"]
    g0 = (lam / (4 * np.pi)) ** 2
    d = np.maximum(np.asarray(d, dtype=float), 1.0)
    return g0 * d ** (-phy["pathloss_exp"])


def snr0(d, phy: dict):
    """Mean receive SNR (linear) at distance d with full AP power and unit beamforming gain."""
    p = dbm_to_w(phy["P_tx_dbm"])
    n = dbm_to_w(phy["noise_dbm"])
    return p * pathloss_gain(d, phy) / n


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def run_dir(layer: str, name: str, seed: int | None = None) -> Path:
    """Create results/<layer>/<name>_<timestamp>[_s<seed>] and write run metadata into it."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{name}_{stamp}" + (f"_s{seed}" if seed is not None else "")
    d = RESULTS_DIR / layer / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_meta(d: Path, **kw) -> None:
    meta = {"git": git_hash(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def latest(layer: str, name: str) -> Path | None:
    """Most recent results/<layer>/<name>_* folder, or None."""
    base = RESULTS_DIR / layer
    if not base.exists():
        return None
    cands = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith(name + "_"))
    return cands[-1] if cands else None
