"""Run every L3 scenario: crypto benchmark, then each scheme x repetition.

  sudo -E ~/cn-venv/bin/python l3_testbed/run_all_l3.py                 # full plan (about 5 x 5 x 11 min)
  sudo -E ~/cn-venv/bin/python l3_testbed/run_all_l3.py --reps 1 --duration 60 --warmup 10   # quick
  sudo -E ~/cn-venv/bin/python l3_testbed/run_all_l3.py --schemes proposed B4_greedy

Trained policies are taken from the latest results/l2/train_<variant>_*_s<seed>/policy.npz
(copy the results/l2 folder from Colab/Drive into this repo first).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from l2_drl.evaluate import find_policy  # noqa: E402

SCHEMES = {
    "proposed":    dict(policy=("npz", "proposed")),
    "B1_agnostic": dict(policy=("npz", "agnostic")),
    "B2_fixed_an": dict(policy="fixed_an"),
    "B3_crypto":   dict(policy="crypto", extra=["--tls"]),
    "B4_greedy":   dict(policy="greedy"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="*", default=list(SCHEMES))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="training seed of the policies to deploy")
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--warmup", type=float, default=60)
    ap.add_argument("--controller", default="ryu", choices=["ryu", "none"])
    ap.add_argument("--edge-cpu", type=float, default=1.0)
    ap.add_argument("--skip-bench", action="store_true")
    a = ap.parse_args()
    py = sys.executable
    script = str(REPO / "l3_testbed" / "run_experiment.py")
    common = ["--controller", a.controller, "--duration", str(a.duration), "--warmup", str(a.warmup),
              "--edge-cpu", str(a.edge_cpu)]
    if not a.skip_bench:
        subprocess.run([py, script, "--crypto-bench", "--controller", a.controller], check=False)
    for scheme in a.schemes:
        spec = SCHEMES[scheme]
        pol = spec["policy"]
        if isinstance(pol, tuple):
            p = find_policy(pol[1], a.seed)
            if p is None:
                print(f"skip {scheme}: no trained policy for variant {pol[1]} seed {a.seed}")
                continue
            pol = str(p)
        for rep in range(a.reps):
            cmd = [py, script, "--scheme", scheme, "--policy", pol, "--rep", str(rep)] + common + spec.get("extra", [])
            print(">>>", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=False)
            subprocess.run(f"{Path(py).parent / 'mn'} -c >/dev/null 2>&1", shell=True)   # clean leftovers between runs


if __name__ == "__main__":
    main()
