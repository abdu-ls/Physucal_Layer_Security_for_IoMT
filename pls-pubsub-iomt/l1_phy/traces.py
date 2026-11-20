"""Link and secrecy traces for trace-driven emulation in the L3 Containernet testbed.

For every topic, grouping mode (single multicast group / split into two groups by distance)
and signal power fraction phi, one channel realisation is drawn per trace step. From it:

link_traces.csv     t, subscriber, topic, mode, phi, delay_ms, loss_pct
    delay_ms = downlink overhead (+ wait for the first group and scheduling delay if split)
               + HARQ retransmissions x dl_retx_s; loss_pct = 100 when all attempts fail.
secrecy_traces.csv  t, topic, mode, phi, secrecy_outage
    1 when any group of the topic is in secrecy outage at that step.

The uplink (device -> AP) has no PLS in this model; the testbed gives it a static netem delay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from l1_phy.channel import sample_group_step


def split_groups(subs, dist):
    """Split subscriber names into (near, far) halves by distance; K = 1 gives one group."""
    order = sorted(subs, key=lambda s: dist[s])
    if len(order) < 2:
        return [order]
    h = len(order) // 2
    return [order[:h], order[h:]]


def generate(params, topics_cfg, rng, out, duration_s=600):
    phy, net = params["phy"], params["net"]
    tb = topics_cfg["testbed"]
    dist = tb["subscribers"]
    step = tb["trace_step_s"]
    T = int(round(duration_s / step))
    M = phy["M_default"]
    A = net["dl_max_attempts"]
    Rmin, Rs = phy["min_decode_rate"], phy["secrecy_rate_target"]
    t = np.round(np.arange(T) * step, 3)
    link_parts, sec_parts = [], []
    for tp in topics_cfg["topics"]:
        for mode in ("single", "split"):
            groups = [list(tp["subscribers"])] if mode == "single" else split_groups(tp["subscribers"], dist)
            for phi in phy["phi_grid"]:
                outage = np.zeros(T, dtype=bool)
                for gi, grp in enumerate(groups):
                    d = [dist[s] for s in grp]
                    K = len(grp)
                    snr, cs = sample_group_step(phi, M, K, d, phy, rng, T * A)
                    snr = snr.reshape(T, A, K)
                    cs = cs.reshape(T, A)
                    outage |= cs[:, 0] < Rs
                    ok = np.log2(1 + snr) >= Rmin                     # (T, A, K)
                    for k, s in enumerate(grp):
                        okk = ok[:, :, k]
                        first = np.where(okk.any(axis=1), okk.argmax(axis=1), -1)
                        lost = first < 0
                        retx = np.where(lost, A - 1, first)
                        base = net["dl_overhead_s"] * (gi + 1) + (net["split_sched_s"] if len(groups) > 1 else 0.0)
                        delay = (base + retx * net["dl_retx_s"]) * 1000.0
                        link_parts.append(pd.DataFrame({
                            "t": t, "subscriber": s, "topic": tp["name"], "mode": mode, "phi": phi,
                            "delay_ms": np.round(delay, 2), "loss_pct": np.where(lost, 100.0, 0.0)}))
                sec_parts.append(pd.DataFrame({"t": t, "topic": tp["name"], "mode": mode, "phi": phi,
                                               "secrecy_outage": outage.astype(int)}))
    links = pd.concat(link_parts, ignore_index=True)
    sec = pd.concat(sec_parts, ignore_index=True)
    links.to_csv(out / "link_traces.csv", index=False)
    sec.to_csv(out / "secrecy_traces.csv", index=False)
    summary = sec.groupby(["topic", "mode", "phi"])["secrecy_outage"].mean().unstack("phi")
    print("  secrecy outage rate per topic/mode/phi:\n" + summary.round(3).to_string())
    return links, sec
