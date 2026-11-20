"""Trace-driven emulation of the AP downlink.

Every trace step (100 ms) and for every subscriber link, it looks up the L1 trace row for
  (subscriber, topic, mode, phi, t)
where topic = the subscriber's highest-priority topic and (mode, phi) = the orchestrator's
current decision for that topic, and applies it with
  tc qdisc replace dev <switch-side interface> root netem delay <d>ms loss <l>%
The switch-side interface lives in the host namespace, so no container capabilities are needed.
Commands are only issued when the values change.
"""
from __future__ import annotations

import subprocess
import threading
import time

import numpy as np
import pandas as pd

PRIO = {"P1": 0, "P2": 1, "P3": 2}


def tc_host(intf, spec):
    cmd = f"tc qdisc replace dev {intf} root netem {spec}"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[replay] tc failed on {intf}: {r.stderr.strip()}")


class TraceReplayer(threading.Thread):
    def __init__(self, link_csv, topics_cfg, sub_intf: dict, orchestrator, step_s=0.1):
        super().__init__(daemon=True)
        df = pd.read_csv(link_csv)
        self.step_s = step_s
        self.n_steps = int(df.t.nunique())
        self.table = {}
        for key, g in df.groupby(["subscriber", "topic", "mode", "phi"]):
            g = g.sort_values("t")
            self.table[(key[0], key[1], key[2], round(float(key[3]), 3))] = (g.delay_ms.values, g.loss_pct.values)
        topics = sorted(topics_cfg["topics"], key=lambda t: (PRIO[t["priority"]], t["deadline_s"]))
        self.sub_topic = {}
        for s in sub_intf:
            for t in topics:
                if s in t["subscribers"]:
                    self.sub_topic[s] = t["name"]
                    break
        self.sub_intf = sub_intf
        self.orch = orchestrator
        self.stop_evt = threading.Event()
        self.last = {}
        self.applied = 0

    def apply_step(self, k):
        dec = self.orch.get() if self.orch is not None else {}
        for s, intf in self.sub_intf.items():
            topic = self.sub_topic.get(s)
            if topic is None:
                continue
            d = dec.get(topic, {"mode": "single", "phi": 1.0})
            key = (s, topic, d["mode"], round(float(d["phi"]), 3))
            if key not in self.table:
                key = (s, topic, "single", 1.0)
            delay, loss = self.table[key]
            val = (float(delay[k % len(delay)]), float(loss[k % len(loss)]))
            if self.last.get(s) != val:
                spec = f"delay {val[0]:.2f}ms" + (f" loss {val[1]:.1f}%" if val[1] > 0 else "")
                tc_host(intf, spec)
                self.last[s] = val
                self.applied += 1

    def run(self):
        k = 0
        nxt = time.time()
        while not self.stop_evt.is_set():
            self.apply_step(k)
            k += 1
            nxt += self.step_s
            self.stop_evt.wait(max(0.0, nxt - time.time()))

    def stop(self):
        self.stop_evt.set()
