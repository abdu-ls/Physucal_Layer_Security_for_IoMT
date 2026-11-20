"""Layer 1 driver: PHY secrecy experiments.

Outputs (in results/l1/l1_<timestamp>/):
  sop_mc.csv      analytical vs Monte Carlo points for figures F1-F3
  phi_opt.csv     optimal signal power fraction phi* per (M, K)          (F4)
  sop_lookup.csv  SOP and COP grid read by the L2 environment
  link_traces.csv, secrecy_traces.csv   replayed by the L3 testbed (see traces.py)

Usage (from the repo root):
  python -m l1_phy.run_l1                 # everything, paper settings
  python -m l1_phy.run_l1 --quick         # fewer trials, for a smoke test
  python -m l1_phy.run_l1 --part lookup   # validate | phiopt | lookup | traces | all
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd

from common import load_params, load_topics, run_dir, write_meta
from l1_phy.channel import simulate
from l1_phy.sop_analytic import sop_k1, sop_semi
from l1_phy import traces


def _analytic(phi, M, K, d, phy, rng, n, **kw):
    if K == 1:
        return sop_k1(phi, M, d, phy, **kw)
    return sop_semi(phi, M, K, d, phy, rng, n=n, **kw)


def validate(phy, rng, trials, n_semi, out):
    rows = []
    t0 = time.time()
    # F1: SOP vs phi, K = 1, several M
    for M in (2, 4, 8):
        for phi in phy["phi_fine"]:
            a = _analytic(phi, M, 1, 25.0, phy, rng, n_semi)
            m = simulate(phi, M, 1, 25.0, phy, rng, trials)
            for meth, r in (("analytic", a), ("mc", m)):
                rows.append(dict(exp="F1", M=M, K=1, phi=phi, eve_density=phy["eve_density"],
                                 guard_radius=phy["guard_radius_m"], dist=25.0, method=meth,
                                 sop=r["sop"], cop=r["cop"]))
    print(f"  F1 done ({time.time() - t0:.0f}s)")
    # F2: SOP vs K, M = 8
    for K in range(1, 7):
        for phi in (0.5, 1.0):
            a = _analytic(phi, 8, K, 22.0, phy, rng, n_semi)
            m = simulate(phi, 8, K, 22.0, phy, rng, trials)
            for meth, r in (("analytic" if K == 1 else "semi", a), ("mc", m)):
                rows.append(dict(exp="F2", M=8, K=K, phi=phi, eve_density=phy["eve_density"],
                                 guard_radius=phy["guard_radius_m"], dist=22.0, method=meth,
                                 sop=r["sop"], cop=r["cop"]))
    print(f"  F2 done ({time.time() - t0:.0f}s)")
    # F3: SOP vs eavesdropper density for several guard radii (M=4, K=2, phi=0.5) and no AN
    for r0, lam, phi in itertools.product(phy["guard_radius_grid"], phy["eve_density_grid"], (0.5, 1.0)):
        a = _analytic(phi, 4, 2, 22.0, phy, rng, n_semi, eve_density=lam, guard_radius=r0)
        m = simulate(phi, 4, 2, 22.0, phy, rng, trials, eve_density=lam, guard_radius=r0)
        for meth, r in (("semi", a), ("mc", m)):
            rows.append(dict(exp="F3", M=4, K=2, phi=phi, eve_density=lam, guard_radius=r0,
                             dist=22.0, method=meth, sop=r["sop"], cop=r["cop"]))
    print(f"  F3 done ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(out / "sop_mc.csv", index=False)
    return df


def phi_opt(phy, rng, n_semi, out, dist=22.0, cop_cap=0.1):
    rows = []
    for M in phy["M_grid"]:
        for K in range(1, M):
            res = [(phi, _analytic(phi, M, K, dist, phy, rng, n_semi)) for phi in phy["phi_fine"]]
            sop = np.array([r["sop"] for _, r in res])
            cop = np.array([r["cop"] for _, r in res])
            phis = np.array([p for p, _ in res])
            i = int(np.argmin(sop))
            ok = np.where(cop <= cop_cap)[0]
            j = int(ok[np.argmin(sop[ok])]) if len(ok) else i
            rows.append(dict(M=M, K=K, dist=dist, phi_star=phis[i], sop_star=sop[i], cop_at_star=cop[i],
                             phi_star_cop_capped=phis[j], sop_capped=sop[j], cop_capped=cop[j],
                             sop_no_an=sop[-1]))
    df = pd.DataFrame(rows)
    df.to_csv(out / "phi_opt.csv", index=False)
    return df


def lookup(phy, rng, n_semi, out, k_max=4):
    rows = []
    grid = list(itertools.product(phy["phi_grid"], phy["M_grid"], range(1, k_max + 1),
                                  phy["eve_density_grid"], phy["dist_grid_m"]))
    t0 = time.time()
    for i, (phi, M, K, lam, d) in enumerate(grid):
        r = _analytic(phi, M, K, d, phy, rng, n_semi, eve_density=lam)
        rows.append(dict(phi=phi, M=M, K=K, eve_density=lam, dist=d, sop=r["sop"], cop=r["cop"]))
        if (i + 1) % 400 == 0:
            print(f"  lookup {i + 1}/{len(grid)} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(out / "sop_lookup.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "validate", "phiopt", "lookup", "traces"])
    ap.add_argument("--quick", action="store_true", help="few trials, short traces (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    params = load_params()
    phy = params["phy"]
    topics = load_topics()
    rng = np.random.default_rng(args.seed)
    trials = 20000 if args.quick else phy["mc_trials"]
    n_semi = 3000 if args.quick else phy["legit_samples_semi"]
    out = run_dir("l1", "l1", args.seed)
    write_meta(out, seed=args.seed, quick=args.quick, part=args.part, phy=phy)
    print(f"L1 results -> {out}")

    if args.part in ("all", "validate"):
        print("Validation (F1-F3) ...")
        validate(phy, rng, trials, n_semi, out)
    if args.part in ("all", "phiopt"):
        print("Optimal phi (F4) ...")
        phi_opt(phy, rng, n_semi, out)
    if args.part in ("all", "lookup"):
        print("Lookup table for L2 ...")
        lookup(phy, rng, n_semi, out)
    if args.part in ("all", "traces"):
        print("Link and secrecy traces for L3 ...")
        dur = 60 if args.quick else topics["testbed"]["trace_duration_s"]
        traces.generate(params, topics, rng, out, duration_s=dur)
    print("L1 done.")


if __name__ == "__main__":
    main()
