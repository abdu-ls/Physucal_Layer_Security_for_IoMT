"""Testbed orchestrator: the SDN control-plane logic that runs the trained policy.

Every decision epoch it builds the policy's observation from a digital twin of the testbed
(the L2 environment configured with the testbed's fixed subscriber distances, eavesdropper
density and load) and writes the per-topic decision to <shared>/decisions.json, which the
publishers and the trace replay read. Decisions are also logged to decisions_log.csv.

Policies: a path to policy.npz (trained), or one of greedy | fixed_an | crypto | random | no_an.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from common import load_params, load_topics                     # noqa: E402
from l2_drl.baselines import BASELINES                          # noqa: E402
from l2_drl.env import IoMTPubSubEnv, decode                    # noqa: E402
from l2_drl.lookup_io import load_lookup                        # noqa: E402
from l2_drl.policy_np import NumpyDuelingPolicy                 # noqa: E402

NAMED = {"greedy": "B4_greedy", "fixed_an": "B2_fixed_an", "crypto": "B3_crypto", "random": "B5_random"}


def no_an_policy(env, obs, mask):
    return BASELINES["B3_crypto"](env, obs, mask)


class Orchestrator(threading.Thread):
    def __init__(self, policy, shared, variant="proposed", load=1.0, lookup=None, seed=0, crypto=False):
        super().__init__(daemon=True)
        self.params, self.topics_cfg = load_params(), load_topics()
        lk, _ = load_lookup(lookup)
        self.env = IoMTPubSubEnv(self.params, self.topics_cfg, lk, variant=variant, crypto=crypto, load=load,
                                 eve_density=self.params["phy"]["eve_density"], seed=seed, train_mode=False)
        if policy.endswith(".npz"):
            self.policy = NumpyDuelingPolicy(policy)
        elif policy == "no_an":
            self.policy = no_an_policy
        else:
            self.policy = BASELINES[NAMED[policy]]
        self.shared = Path(shared)
        self.stop_evt = threading.Event()
        self.current = {}
        self.lock = threading.Lock()
        self.log = open(self.shared / "decisions_log.csv", "w")
        self.log.write("t,epoch,topic,tier,mode,phi\n")
        self._reset_twin()

    def _reset_twin(self):
        env = self.env
        env.reset()
        env.dist = dict(self.topics_cfg["testbed"]["subscribers"])      # fixed testbed distances
        for t in env.topics:
            subs = sorted(t["subscribers"], key=lambda s: env.dist[s])
            split = [subs[: len(subs) // 2], subs[len(subs) // 2:]] if len(subs) > 1 else [subs]
            env.groups[t["name"]] = ([subs], split)

    def decide_epoch(self):
        env = self.env
        out = {}
        obs = env._obs()
        for _ in range(env.n_topics):
            t = env.current_topic
            a = self.policy(env, obs, env.action_mask())
            tier, grp, pi = decode(a)
            out[t["name"]] = {"tier": int(tier), "mode": "split" if grp else "single", "phi": float(env.phi_grid[pi])}
            obs, _, done, _ = env.step(a)
            if done:
                self._reset_twin()
                obs = env._obs()
        return out

    def write(self, dec, epoch):
        tmp = self.shared / "decisions.json.tmp"
        with open(tmp, "w") as f:
            json.dump(dec, f)
        os.replace(tmp, self.shared / "decisions.json")
        now = time.time()
        for k, v in dec.items():
            self.log.write(f"{now:.3f},{epoch},{k},{v['tier']},{v['mode']},{v['phi']}\n")
        self.log.flush()

    def run(self):
        epoch_s = self.params["net"]["epoch_s"]
        epoch = 0
        nxt = time.time()
        while not self.stop_evt.is_set():
            dec = self.decide_epoch()
            with self.lock:
                self.current = dec
            self.write(dec, epoch)
            epoch += 1
            nxt += epoch_s
            self.stop_evt.wait(max(0.0, nxt - time.time()))
        self.log.close()

    def get(self):
        with self.lock:
            return dict(self.current)

    def stop(self):
        self.stop_evt.set()
