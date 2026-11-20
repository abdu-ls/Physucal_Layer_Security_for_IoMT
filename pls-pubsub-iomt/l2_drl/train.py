"""Train the PLS-aware Dueling DQN orchestrator (or one of its variants).

Usage (from the repo root):
  python -m l2_drl.train --variant proposed --seed 0
  python -m l2_drl.train --variant agnostic --seed 0          # B1: security-agnostic DQN
  python -m l2_drl.train --variant no_sdn --seed 0            # ablations: no_priority | no_sensitivity | no_sdn | no_an
  python -m l2_drl.train --variant proposed --beta-s 3 --seed 0   # beta_s sweep (F9)
  add --quick for a short smoke run.

Writes results/l2/train_<variant>[_b<beta>]_<timestamp>_s<seed>/ with
  model.pt, policy.npz (NumPy export), train_log.csv, meta.json
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from common import load_params, load_topics, run_dir, write_meta
from l2_drl.dueling_agent import DuelingDQNAgent
from l2_drl.env import N_ACTIONS, OBS_DIM, VARIANTS, IoMTPubSubEnv
from l2_drl.lookup_io import load_lookup


def train(variant, seed, episodes=None, beta_s=None, lookup_path=None, quick=False):
    params, topics = load_params(), load_topics()
    cfg = dict(params["drl"])
    if quick:
        cfg.update(warmup_steps=500, eps_decay_steps=5000, target_update_steps=500)
    episodes = episodes or (20 if quick else cfg["train_episodes"])
    lookup, lk_src = load_lookup(lookup_path)
    env = IoMTPubSubEnv(params, topics, lookup, variant=variant, beta_s=beta_s, seed=seed)
    agent = DuelingDQNAgent(OBS_DIM, N_ACTIONS, cfg, seed=seed)
    name = f"train_{variant}" + (f"_b{beta_s:g}" if beta_s is not None else "")
    out = run_dir("l2", name, seed)
    write_meta(out, variant=variant, seed=seed, episodes=episodes, beta_s=env.beta_s, lookup=str(lk_src),
               drl=cfg, reward=params["reward"])
    print(f"[{variant}] seed {seed}: {episodes} episodes -> {out}")
    log = []
    t0 = time.time()
    for ep in range(episodes):
        obs = env.reset(seed=seed * 100000 + ep)
        mask = env.action_mask()
        done = False
        ret, n_msg, lat_sum, miss_sum, viol_w, losses = 0.0, 0, 0.0, 0, 0.0, []
        while not done:
            a = agent.act(obs, mask)
            obs2, r, done, info = env.step(a)
            mask2 = env.action_mask() if not done else np.ones(N_ACTIONS, bool)
            o2 = obs2 if obs2 is not None else np.zeros(OBS_DIM, np.float32)
            agent.buf.add(obs, a, r, o2, mask2, float(done))
            l = agent.update()
            if l is not None:
                losses.append(l)
            ret += r
            if info["n"] > 0:
                n_msg += info["n"]
                lat_sum += info["lat"].sum()
                miss_sum += int(info["miss"].sum())
                viol_w += info["viol"] * info["n"]
            obs, mask = o2, mask2
        log.append(dict(episode=ep, ret=ret, mean_lat=lat_sum / max(n_msg, 1), miss_rate=miss_sum / max(n_msg, 1),
                        viol_rate=viol_w / max(n_msg, 1), eps=agent.epsilon(),
                        loss=float(np.mean(losses)) if losses else np.nan, load=env.load, eve_density=env.lam))
        if (ep + 1) % max(1, episodes // 20) == 0 or ep == episodes - 1:
            last = pd.DataFrame(log[-max(1, episodes // 20):]).mean()
            print(f"  ep {ep + 1:4d}  return {last.ret:8.1f}  lat {last.mean_lat * 1000:6.1f} ms  "
                  f"miss {last.miss_rate:.3f}  viol {last.viol_rate:.3f}  eps {agent.epsilon():.2f}  "
                  f"({time.time() - t0:.0f}s)")
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    agent.save(out / "model.pt")
    agent.export_npz(out / "policy.npz")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="proposed", choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--beta-s", type=float, default=None)
    ap.add_argument("--lookup", default=None, help="path to sop_lookup.csv (default: latest L1 run)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    train(a.variant, a.seed, a.episodes, a.beta_s, a.lookup, a.quick)


if __name__ == "__main__":
    main()
