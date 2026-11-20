"""Run every L2 training job, then the evaluation.

  python -m l2_drl.run_l2_all                       # all variants x all seeds + beta sweep + eval
  python -m l2_drl.run_l2_all --quick               # smoke test (short runs, seed 0)
  python -m l2_drl.run_l2_all --variants proposed agnostic --seeds 0 1
  python -m l2_drl.run_l2_all --skip-train          # evaluation only
  python -m l2_drl.run_l2_all --no-eval             # training only

Training jobs are independent, so on Colab you can split them across sessions with
--variants / --seeds; results accumulate in the same results/ folder.
"""
from __future__ import annotations

import argparse

from common import load_params
from l2_drl import evaluate
from l2_drl.env import VARIANTS
from l2_drl.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--beta-seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--no-beta-sweep", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--no-eval", action="store_true", help="train only")
    a = ap.parse_args()
    cfg = load_params()["drl"]
    seeds = a.seeds if a.seeds is not None else ([0] if a.quick else cfg["seeds"])
    if not a.skip_train:
        for v in a.variants:
            for s in seeds:
                train(v, s, quick=a.quick)
        if not a.no_beta_sweep and "proposed" in a.variants:
            for beta in cfg["beta_s_grid"]:
                for s in ([0] if a.quick else a.beta_seeds):
                    train("proposed", s, beta_s=beta, quick=a.quick)
    if a.no_eval:
        return
    import sys
    sys.argv = [sys.argv[0]] + (["--quick"] if a.quick else [])
    evaluate.main()


if __name__ == "__main__":
    main()
