"""Non-learning policies. Each is a callable policy(env, obs, mask) -> action index."""
from __future__ import annotations

import numpy as np

from l2_drl.env import N_ACTIONS, decode, encode


def _phi_index(env, phi):
    return int(np.argmin(np.abs(env.phi_grid - phi)))


def _best_tier(env, group, phi_idx, mask):
    """Tier with the lower predicted latency for a fixed grouping and phi."""
    cands = [encode(t, group, phi_idx) for t in (0, 1) if mask[encode(t, group, phi_idx)]]
    return min(cands, key=lambda a: env.predict(a)["latency"])


def random_policy(env, obs, mask):
    return int(env.rng.choice(np.flatnonzero(mask)))


def fixed_an_policy(env, obs, mask, phi=0.5):
    """B2: phi = 0.5 for every topic (or no AN when impossible), one group, latency-greedy tier."""
    pi = _phi_index(env, phi)
    if not mask[encode(0, 0, pi)] and not mask[encode(1, 0, pi)]:
        pi = _phi_index(env, 1.0)
    return _best_tier(env, 0, pi, mask)


def greedy_policy(env, obs, mask):
    """B4: one-step model-based. Among actions whose predicted SOP meets the topic's target,
    take the lowest predicted latency (+ loss penalty); if none meets it, take the lowest SOP."""
    t = env.current_topic
    preds = {a: env.predict(a) for a in np.flatnonzero(mask)}
    ok = [a for a, p in preds.items() if p["sop"] <= t["epsilon"]]
    score = lambda a: preds[a]["latency"] + t["deadline_s"] * (preds[a]["p_lost"] * 5 + preds[a]["late"])
    if ok:
        return int(min(ok, key=score))
    return int(min(preds, key=lambda a: (round(preds[a]["sop"], 3), score(a))))


def crypto_policy(env, obs, mask):
    """B3: no AN (confidentiality from TLS/AES), one group, latency-greedy tier.
    Use with an environment created with crypto=True so the overheads are charged."""
    return _best_tier(env, 0, _phi_index(env, 1.0), mask)


BASELINES = {
    "B2_fixed_an": fixed_an_policy,
    "B3_crypto": crypto_policy,
    "B4_greedy": greedy_policy,
    "B5_random": random_policy,
}
