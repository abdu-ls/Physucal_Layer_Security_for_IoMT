"""Evaluate trained policies and baselines on identical episodes.

Trained policies are read from the latest results/l2/train_<variant>_*_s<seed>/policy.npz
(NumPy inference, PyTorch not needed). Missing trained runs are skipped with a warning.

Outputs (results/l2/eval_<timestamp>/):
  eval.csv         per scheme, seed, load, eve density, topic: latency, deadline miss, SOP, energy ...
  ablation.csv     proposed + ablation variants at load 1.0                   (F10)
  sweep_beta.csv   proposed trained with different beta_s, at load 1.0        (F9)

Usage:
  python -m l2_drl.evaluate                 # full evaluation
  python -m l2_drl.evaluate --quick         # 2 episodes, seed 0, load 1.0
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS_DIR, load_params, load_topics, run_dir, write_meta
from l2_drl.baselines import BASELINES
from l2_drl.env import IoMTPubSubEnv
from l2_drl.lookup_io import load_lookup
from l2_drl.policy_np import NumpyDuelingPolicy

TRAINED = {"proposed": "proposed", "B1_agnostic": "agnostic", "A_no_priority": "no_priority",
           "A_no_sensitivity": "no_sensitivity", "A_no_sdn": "no_sdn", "A_no_an": "no_an"}
ABLATIONS = ["proposed", "A_no_priority", "A_no_sensitivity", "A_no_sdn", "A_no_an"]


def find_policy(variant, seed, beta=None):
    base = RESULTS_DIR / "l2"
    name = f"train_{variant}" + (f"_b{beta:g}" if beta is not None else "")
    pat = re.compile(rf"^{re.escape(name)}_\d{{8}}-\d{{6}}_s{seed}$")
    cands = sorted(p for p in base.glob(f"{name}_*") if pat.match(p.name)) if base.exists() else []
    return (cands[-1] / "policy.npz") if cands and (cands[-1] / "policy.npz").exists() else None


def run_episodes(env, policy, seeds, tag):
    acc = {}
    for s in seeds:
        obs = env.reset(seed=s)
        done = False
        while not done:
            mask = env.action_mask()
            a = policy(env, obs, mask)
            obs, _, done, info = env.step(a)
            if info["n"] == 0:
                continue
            d = acc.setdefault(info["topic"], dict(priority=info["priority"], sensitivity=info["sensitivity"],
                                                   lat=[], miss=0, lost=0, n=0, sop_w=0.0, viol_w=0.0, sec=0,
                                                   energy=0.0, cloud=0, split=0, phi_w=0.0))
            n = info["n"]
            d["lat"].append(info["lat"])
            d["miss"] += int(info["miss"].sum())
            d["lost"] += int(info["lost"].sum())
            d["n"] += n
            d["sop_w"] += info["sop"] * n
            d["viol_w"] += info["viol"] * n
            d["sec"] += int(info["sec_out"].sum())
            d["energy"] += float(info["energy"].sum())
            d["cloud"] += n * info["tier"]
            d["split"] += n * info["group"]
            d["phi_w"] += n * info["phi"]
    rows = []
    for topic, d in acc.items():
        lat = np.concatenate(d["lat"])
        n = d["n"]
        rows.append(dict(**tag, topic=topic, priority=d["priority"], sensitivity=d["sensitivity"], n=n,
                         mean_lat=lat.mean(), p95_lat=np.percentile(lat, 95), miss_rate=d["miss"] / n,
                         lost_rate=d["lost"] / n, sop=d["sop_w"] / n, viol_rate=d["viol_w"] / n,
                         sec_out_rate=d["sec"] / n, energy_per_msg=d["energy"] / n,
                         share_cloud=d["cloud"] / n, share_split=d["split"] / n, mean_phi=d["phi_w"] / n))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lookup", default=None)
    ap.add_argument("--schemes", nargs="*", default=None, help="subset of scheme names")
    a = ap.parse_args()

    params, topics = load_params(), load_topics()
    cfg = params["drl"]
    lookup, lk = load_lookup(a.lookup)
    seeds = [0] if a.quick else cfg["seeds"]
    loads = [1.0] if a.quick else cfg["load_levels"]
    n_ep = 2 if a.quick else cfg["eval_episodes"]
    lam_default = params["phy"]["eve_density"]
    lam_sweep = [] if a.quick else params["phy"]["eve_density_grid"]
    out = run_dir("l2", "eval")
    write_meta(out, seeds=seeds, loads=loads, episodes=n_ep, lookup=str(lk))
    print(f"Evaluation -> {out}")

    schemes = list(TRAINED) + list(BASELINES)
    if a.schemes:
        schemes = [s for s in schemes if s in a.schemes]
    rows = []
    for scheme in schemes:
        for seed in seeds:
            if scheme in TRAINED:
                variant = TRAINED[scheme]
                pol_path = find_policy(variant, seed)
                if pol_path is None:
                    print(f"  skip {scheme} seed {seed}: no trained policy found")
                    continue
                policy = NumpyDuelingPolicy(pol_path)
            else:
                variant, policy = "proposed", BASELINES[scheme]
            crypto = scheme == "B3_crypto"
            ep_seeds = [10_000_000 + seed * 1000 + e for e in range(n_ep)]
            conds = [(L, lam_default) for L in loads]
            if scheme in ("proposed", "B1_agnostic", "B4_greedy", "B2_fixed_an"):
                conds += [(1.0, lam) for lam in lam_sweep if lam != lam_default]
            for load, lam in conds:
                env = IoMTPubSubEnv(params, topics, lookup, variant=variant, crypto=crypto, load=load,
                                    eve_density=lam, seed=seed, train_mode=False)
                rows += run_episodes(env, policy, ep_seeds,
                                     dict(scheme=scheme, seed=seed, load=load, eve_density=lam,
                                          protected_by_crypto=int(crypto)))
            print(f"  {scheme} seed {seed} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "eval.csv", index=False)
    if len(df):
        base = df[(df.load == 1.0) & (df.eve_density == lam_default)]
        base[base.scheme.isin(ABLATIONS)].to_csv(out / "ablation.csv", index=False)

    # beta_s sweep (F9)
    sweep = []
    for beta in cfg["beta_s_grid"]:
        for seed in seeds:
            p = find_policy("proposed", seed, beta)
            if p is None:
                continue
            env = IoMTPubSubEnv(params, topics, lookup, variant="proposed", load=1.0, eve_density=lam_default,
                                seed=seed, train_mode=False)
            ep_seeds = [10_000_000 + seed * 1000 + e for e in range(n_ep)]
            for r in run_episodes(env, NumpyDuelingPolicy(p), ep_seeds, dict(scheme="proposed", beta_s=beta, seed=seed)):
                sweep.append(r)
    if sweep:
        pd.DataFrame(sweep).to_csv(out / "sweep_beta.csv", index=False)
    print("Evaluation done.")
    return out


if __name__ == "__main__":
    main()
