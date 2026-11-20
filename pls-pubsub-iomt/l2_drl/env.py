"""PLS-aware topic orchestration environment for IoMT pub/sub (numpy only, no gym dependency).

One episode = `episode_epochs` decision epochs. In every epoch the orchestrator makes one
decision per topic, in priority order (P1 first, then by deadline). A decision fixes, for all
messages of that topic in the epoch:
    tier      : edge broker (0) or cloud broker (1)
    grouping  : one multicast group (0) or two groups split by distance (1)
    phi       : signal power fraction from phi_grid (1.0 = no artificial noise)
Action index a = tier * 10 + grouping * 5 + phi_index   (20 actions).

Per-message latency = uplink + broker queueing (+ 2x backhaul for cloud) + downlink
(group overhead, split scheduling, HARQ retransmissions, AP airtime contention).
SOP and connection outage (COP) per group come from the L1 lookup table.

Variants (ablations / baselines) change what the agent observes and how it is rewarded;
the metrics are always computed with the true priorities and sensitivity targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOPIC_ORDER_KEY = {"P1": 0, "P2": 1, "P3": 2}
PRIO_FEAT = {"P1": 1.0, "P2": 0.5, "P3": 0.0}
SENS_FEAT = {"High": 1.0, "Medium": 0.5, "Low": 0.0}
N_ACTIONS = 20
OBS_DIM = 24

VARIANTS = {
    "proposed":       dict(),
    "agnostic":       dict(secrecy_aware=False, allow_an=False),       # B1: previous framework
    "no_priority":    dict(use_priority=False),
    "no_sensitivity": dict(use_sensitivity=False),
    "no_sdn":         dict(use_sdn=False),
    "no_an":          dict(allow_an=False),
}


def decode(a: int):
    return a // 10, (a // 5) % 2, a % 5


def encode(tier: int, group: int, phi_idx: int) -> int:
    return tier * 10 + group * 5 + phi_idx


class Lookup:
    """SOP/COP lookup on the L1 grid; distances are rounded UP to the next grid point (conservative)."""

    def __init__(self, df: pd.DataFrame):
        self.phis = np.sort(df.phi.unique())
        self.Ms = np.sort(df.M.unique())
        self.Ks = np.sort(df.K.unique())
        self.lams = np.sort(df.eve_density.unique())
        self.ds = np.sort(df.dist.unique())
        shape = (len(self.phis), len(self.Ms), len(self.Ks), len(self.lams), len(self.ds))
        self.sop = np.full(shape, np.nan)
        self.cop = np.full(shape, np.nan)
        idx = (np.searchsorted(self.phis, df.phi), np.searchsorted(self.Ms, df.M), np.searchsorted(self.Ks, df.K),
               np.searchsorted(self.lams, df.eve_density), np.searchsorted(self.ds, df.dist))
        self.sop[idx] = df.sop.values
        self.cop[idx] = df.cop.values

    def __call__(self, phi, M, K, lam, d):
        i = int(np.argmin(np.abs(self.phis - phi)))
        j = int(np.argmin(np.abs(self.Ms - M)))
        k = int(np.clip(np.searchsorted(self.Ks, K), 0, len(self.Ks) - 1))
        l = int(np.argmin(np.abs(np.log10(self.lams) - np.log10(lam))))
        m = int(np.clip(np.searchsorted(self.ds, d - 1e-9), 0, len(self.ds) - 1))
        return float(self.sop[i, j, k, l, m]), float(self.cop[i, j, k, l, m])


class IoMTPubSubEnv:
    def __init__(self, params, topics_cfg, lookup: Lookup, variant="proposed", crypto=False,
                 load=None, M=None, eve_density=None, beta_s=None, seed=0, train_mode=True):
        self.p, self.net, self.phy, self.rw = params, params["net"], params["phy"], params["reward"]
        self.topics = sorted(topics_cfg["topics"], key=lambda t: (TOPIC_ORDER_KEY[t["priority"]], t["deadline_s"]))
        self.roles = list(topics_cfg["testbed"]["subscribers"].keys())
        self.lookup = lookup
        self.variant = variant
        flags = dict(use_priority=True, use_sensitivity=True, use_sdn=True, allow_an=True, secrecy_aware=True)
        flags.update(VARIANTS[variant])
        self.__dict__.update(flags)
        self.crypto = crypto
        self.fixed_load, self.fixed_lam = load, eve_density
        self.M = M or self.phy["M_default"]
        self.beta_s = self.rw["beta_s"] if beta_s is None else beta_s
        if not self.secrecy_aware:
            self.beta_s = 0.0
        self.phi_grid = np.array(self.phy["phi_grid"])
        self.train_mode = train_mode
        self.rng = np.random.default_rng(seed)
        self.n_topics = len(self.topics)

    # ------------------------------------------------------------------ episode
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        self.load = self.fixed_load if self.fixed_load is not None else r.uniform(0.5, 1.25)
        self.lam = self.fixed_lam if self.fixed_lam is not None else float(r.choice(self.phy["eve_density_grid"]))
        lo, hi = 3.0, self.phy["ward_radius_m"]
        self.dist = {s: r.uniform(lo, hi) for s in self.roles}
        self.groups = {}
        for t in self.topics:
            subs = sorted(t["subscribers"], key=lambda s: self.dist[s])
            single = [subs]
            split = [subs[: len(subs) // 2], subs[len(subs) // 2:]] if len(subs) > 1 else [subs]
            self.groups[t["name"]] = (single, split)
        self.epoch, self.j = 0, 0
        self.backlog = {"edge": 0.0, "cloud": 0.0}
        self.airtime, self.rho = 0.0, 0.0
        self.last_total = 0.0
        self.last_miss = np.zeros(self.n_topics)
        self.last_viol = np.zeros(self.n_topics)
        self._draw_arrivals()
        return self._obs()

    def _draw_arrivals(self):
        e = self.net["epoch_s"]
        lam = np.array([t["rate_per_device"] for t in self.topics]) * self.net["n_devices"] * self.load * e
        self.arrivals = self.rng.poisson(lam)

    # ------------------------------------------------------------- observation
    def _obs(self):
        t = self.topics[self.j]
        single, split = self.groups[t["name"]]
        maxd = lambda g: max(self.dist[s] for s in g) / self.phy["ward_radius_m"]
        o = np.zeros(OBS_DIM, dtype=np.float32)
        o[0] = PRIO_FEAT[t["priority"]] if self.use_priority else 0.0
        if self.use_sensitivity and self.secrecy_aware:
            o[1] = SENS_FEAT[t["sensitivity"]]
            o[2] = -np.log10(t["epsilon"]) / 2.0
        o[3] = (np.log10(t["deadline_s"]) + 1.0) / 2.0
        o[4] = self.arrivals[self.j] / 150.0
        o[5] = t["K"] / 4.0
        o[6] = maxd(single[0])
        o[7] = maxd(split[0])
        o[8] = maxd(split[-1])
        if self.use_sdn:
            o[9] = min(self.backlog["edge"] / self.net["edge_service_rate"], 2.0)
            o[10] = min(self.backlog["cloud"] / self.net["cloud_service_rate"], 2.0)
            o[11] = self.rho
            o[17] = self.last_total / self.net["edge_service_rate"]
        if self.secrecy_aware:
            o[12] = (np.log10(self.lam) + 4.0) / 2.0
        o[13] = self.M / 8.0
        o[14] = self.last_miss[self.j]
        o[15] = self.last_viol[self.j]
        o[16] = self.epoch / self.net["episode_epochs"]
        o[18 + self.j] = 1.0
        return o

    def action_mask(self):
        t = self.topics[self.j]
        m = np.ones(N_ACTIONS, dtype=bool)
        for a in range(N_ACTIONS):
            tier, grp, pi = decode(a)
            if grp == 1 and t["K"] == 1:
                m[a] = False
            if self.phi_grid[pi] < 1.0:
                groups = self.groups[t["name"]][grp]
                if not self.allow_an or all(len(g) >= self.M for g in groups):
                    m[a] = False
        return m

    @property
    def current_topic(self):
        return self.topics[self.j]

    # -------------------------------------------------------- physics of a decision
    def _group_stats(self, t, grp, phi):
        out = []
        for g in self.groups[t["name"]][grp]:
            K = len(g)
            ph = phi if K < self.M else 1.0
            sop, cop = self.lookup(ph, self.M, K, self.lam, max(self.dist[s] for s in g))
            out.append((sop, cop))
        return out

    def predict(self, a):
        """Expected latency, deadline-miss proxy and SOP for action a on the current topic (for model-based baselines)."""
        t = self.topics[self.j]
        tier, grp, pi = decode(a)
        phi = self.phi_grid[pi]
        n = self.arrivals[self.j]
        net = self.net
        size = t["payload_bytes"] + (self.p["crypto"]["overhead_bytes"] if self.crypto else 0)
        up = size * 8 / net["uplink_rate_bps"] + np.mean(net["uplink_access_s"])
        if tier == 0:
            wait = (self.backlog["edge"] + n / 2 + 1) / net["edge_service_rate"]
        else:
            wait = (self.backlog["cloud"] + n / 2 + 1) / net["cloud_service_rate"] + 2 * net["backhaul_delay_s"]
        stats = self._group_stats(t, grp, phi)
        ng = len(stats)
        A = net["dl_max_attempts"]
        dl = 0.0
        p_lost = 0.0
        for gi, (_, cop) in enumerate(stats):
            exp_retx = sum(cop ** k for k in range(1, A))
            dl = max(dl, net["dl_overhead_s"] * (gi + 1) + exp_retx * net["dl_retx_s"])
            p_lost = 1 - (1 - p_lost) * (1 - cop ** A)
        dl += (net["split_sched_s"] if ng > 1 else 0.0) + net["dl_overhead_s"] * self.rho / max(1 - self.rho, 0.05)
        crypto = self._crypto_delay() if self.crypto else 0.0
        lat = up + wait + dl + crypto
        sop = 1 - np.prod([1 - s for s, _ in stats])
        return dict(latency=lat, p_lost=p_lost, sop=sop, late=lat > t["deadline_s"])

    def _crypto_delay(self):
        c = self.p["crypto"]
        return 2 * c["per_msg_cpu_s"] + c["handshake_s"] / c["handshake_every"]

    def step(self, a):
        t = self.topics[self.j]
        tier, grp, pi = decode(int(a))
        phi = float(self.phi_grid[pi])
        net, r = self.net, self.rng
        n = int(self.arrivals[self.j])
        info = dict(topic=t["name"], priority=t["priority"], sensitivity=t["sensitivity"], n=n,
                    tier=tier, group=grp, phi=phi, epoch=self.epoch)
        stats = self._group_stats(t, grp, phi)
        sop = 1 - np.prod([1 - s for s, _ in stats])
        info["sop"] = sop
        info["viol"] = float(sop > t["epsilon"])
        if n > 0:
            size = t["payload_bytes"] + (self.p["crypto"]["overhead_bytes"] if self.crypto else 0)
            up_tx = size * 8 / net["uplink_rate_bps"]
            up = up_tx + r.uniform(*net["uplink_access_s"], size=n)
            key = "edge" if tier == 0 else "cloud"
            mu = net["edge_service_rate"] if tier == 0 else net["cloud_service_rate"]
            order = np.arange(n) + r.uniform(0, 1, size=n)
            wait = (self.backlog[key] + order) / mu + r.exponential(1.0 / mu, size=n)
            if tier == 1:
                wait = wait + 2 * net["backhaul_delay_s"]
            self.backlog[key] += n
            A = net["dl_max_attempts"]
            dl = np.zeros(n)
            lost = np.zeros(n, dtype=bool)
            tx_count = np.zeros(n)
            for gi, (_, cop) in enumerate(stats):
                fails = r.random((n, A)) < cop
                ok = ~fails
                first = np.where(ok.any(axis=1), ok.argmax(axis=1), A - 1)
                lost |= ~ok.any(axis=1)
                tx_count += first + 1
                dl = np.maximum(dl, net["dl_overhead_s"] * (gi + 1) + first * net["dl_retx_s"])
            if len(stats) > 1:
                dl += net["split_sched_s"]
            dl += net["dl_overhead_s"] * self.rho / max(1 - self.rho, 0.05) * r.exponential(1.0, size=n)
            self.airtime += tx_count.sum() * net["dl_overhead_s"]
            lat = up + wait + dl
            e_dev = net["device_tx_power_w"] * up_tx * np.ones(n)
            e_ap = net["ap_power_w"] * tx_count * net["dl_overhead_s"]
            if self.crypto:
                c = self.p["crypto"]
                lat = lat + self._crypto_delay()
                e_dev = e_dev + net["device_cpu_power_w"] * (c["per_msg_cpu_s"] + c["handshake_s"] / c["handshake_every"])
            energy = e_dev + e_ap
            miss = (lat > t["deadline_s"]) | lost
            sec_out = r.random(n) < sop
            info.update(lat=lat, lost=lost, miss=miss, energy=energy, sec_out=sec_out)
            # ---- reward (agent's view: variant may hide priority/sensitivity)
            w_p = self.rw["priority_weight"][t["priority"]] if self.use_priority else 1.0
            w_s = self.rw["sensitivity_weight"][t["sensitivity"]] if self.use_sensitivity else 1.0
            eps = t["epsilon"] if self.use_sensitivity else 0.05
            rew = -(self.rw["w_latency"] * np.mean(np.minimum(lat / t["deadline_s"], 5.0))
                    + self.rw["w_energy"] * np.mean(energy) / self.rw["energy_ref_j"])
            rew -= self.rw["beta_p"] * w_p * miss.mean()
            rew -= self.beta_s * w_s * max(0.0, sop - eps)
            self.last_miss[self.j] = miss.mean()
        else:
            rew = 0.0
            self.last_miss[self.j] = 0.0
        self.last_viol[self.j] = info["viol"]
        info["reward"] = rew
        # advance
        self.j += 1
        done = False
        if self.j >= self.n_topics:
            self._end_epoch()
            done = self.epoch >= self.net["episode_epochs"]
        obs = None if done else self._obs()
        return obs, float(rew), done, info

    def _end_epoch(self):
        e = self.net["epoch_s"]
        self.backlog["edge"] = max(0.0, self.backlog["edge"] - self.net["edge_service_rate"] * e)
        self.backlog["cloud"] = max(0.0, self.backlog["cloud"] - self.net["cloud_service_rate"] * e)
        self.rho = min(self.airtime / e, 0.95)
        self.airtime = 0.0
        self.last_total = float(self.arrivals.sum())
        self.epoch += 1
        self.j = 0
        self._draw_arrivals()
