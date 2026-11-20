"""Monte Carlo channel model: AN-aided multicast from an M-antenna AP to K subscribers,
with passive single-antenna eavesdroppers forming a PPP in an annulus around the AP.

Conventions
-----------
* h_i, g_e in C^M, CN(0, I) small-scale fading (Rayleigh), times sqrt(path-loss gain).
* Received signal gain of subscriber i: |h_i^H w|^2.
* Signal beamformer w = normalise(sum_i h_i/||h_i||)  (MRT when K = 1).
* Artificial noise spans Z, an orthonormal basis of the null space of [h_1 ... h_K]^H,
  so subscribers see no AN. Needs M > K; otherwise AN is switched off (phi forced to 1).
* phi = fraction of power on the signal; (1-phi) on AN, spread evenly over the M-K dims.
"""
from __future__ import annotations

import numpy as np

from common import dbm_to_w, pathloss_gain


def cn(shape, rng):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def beamformer_and_null(H):
    """H: (B, K, M) small-scale channels (rows h_i). Returns w (B, M), Z (B, M, M-K)."""
    B, K, M = H.shape
    hn = H / np.linalg.norm(H, axis=2, keepdims=True)
    w = hn.sum(axis=1)
    w = w / np.linalg.norm(w, axis=1, keepdims=True)
    if M > K:
        # rows of A = H^* give h_i^H x; null space of A = right singular vectors K..M-1
        _, _, vh = np.linalg.svd(H.conj(), full_matrices=True)
        Z = vh[:, K:, :].conj().transpose(0, 2, 1)       # (B, M, M-K)
    else:
        Z = np.zeros((B, M, 0), dtype=complex)
    return w, Z


def effective_phi(phi: float, M: int, K: int) -> float:
    return phi if M > K else 1.0


def sample_eves(B, phy, rng, eve_density=None, guard_radius=None):
    """PPP eavesdroppers in annulus [r0, R]. Returns distances (B, Nmax) and mask (B, Nmax)."""
    lam = phy["eve_density"] if eve_density is None else eve_density
    r0 = phy["guard_radius_m"] if guard_radius is None else guard_radius
    R = phy["eve_outer_radius_m"]
    area = np.pi * (R ** 2 - r0 ** 2)
    n = rng.poisson(lam * area, size=B)
    nmax = max(int(n.max()), 1)
    u = rng.uniform(r0 ** 2, R ** 2, size=(B, nmax))
    r = np.sqrt(u)
    mask = np.arange(nmax)[None, :] < n[:, None]
    return r, mask


def simulate(phi, M, K, sub_dist, phy, rng, trials, eve_density=None, guard_radius=None,
             batch=20000):
    """Monte Carlo SOP and connection-outage probability (COP) for one multicast group.

    sub_dist: scalar or length-K array of subscriber distances (m).
    Returns dict with sop, cop, mean secrecy capacity (bit/s/Hz).
    """
    sub_dist = np.broadcast_to(np.asarray(sub_dist, dtype=float), (K,))
    phi_e = effective_phi(phi, M, K)
    P = dbm_to_w(phy["P_tx_dbm"])
    N0 = dbm_to_w(phy["noise_dbm"])
    Rs, Rmin = phy["secrecy_rate_target"], phy["min_decode_rate"]
    beta_sub = pathloss_gain(sub_dist, phy)          # (K,)
    n_out = n_cop = 0
    cs_sum = 0.0
    done = 0
    while done < trials:
        b = min(batch, trials - done)
        H = cn((b, K, M), rng)
        w, Z = beamformer_and_null(H)
        gain = np.abs(np.einsum("bkm,bm->bk", H.conj(), w)) ** 2      # (b, K)
        g_sub = phi_e * P * beta_sub[None, :] * gain / N0
        gmin = g_sub.min(axis=1)
        r, mask = sample_eves(b, phy, rng, eve_density, guard_radius)
        G = cn((b, r.shape[1], M), rng)
        beta_e = pathloss_gain(r, phy)
        sig = np.abs(np.einsum("bnm,bm->bn", G.conj(), w)) ** 2
        if Z.shape[2] > 0 and phi_e < 1.0:
            an = (np.abs(np.einsum("bnm,bmk->bnk", G.conj(), Z)) ** 2).sum(axis=2)
            an_pow = (1 - phi_e) * P * beta_e * an / (M - K)
        else:
            an_pow = 0.0
        g_eve = phi_e * P * beta_e * sig / (an_pow + N0)
        g_eve = np.where(mask, g_eve, 0.0).max(axis=1)
        cs = np.maximum(np.log2(1 + gmin) - np.log2(1 + g_eve), 0.0)
        n_out += int((cs < Rs).sum())
        n_cop += int((np.log2(1 + gmin) < Rmin).sum())
        cs_sum += float(cs.sum())
        done += b
    return {"sop": n_out / trials, "cop": n_cop / trials, "mean_cs": cs_sum / trials}


def sample_group_step(phi, M, K, sub_dist, phy, rng, T, eve_density=None, guard_radius=None):
    """One channel draw per time step (used for link traces).
    Returns per-subscriber SNR (T, K) and secrecy capacity (T,)."""
    sub_dist = np.asarray(sub_dist, dtype=float)
    phi_e = effective_phi(phi, M, K)
    P = dbm_to_w(phy["P_tx_dbm"])
    N0 = dbm_to_w(phy["noise_dbm"])
    H = cn((T, K, M), rng)
    w, Z = beamformer_and_null(H)
    gain = np.abs(np.einsum("bkm,bm->bk", H.conj(), w)) ** 2
    g_sub = phi_e * P * pathloss_gain(sub_dist, phy)[None, :] * gain / N0
    r, mask = sample_eves(T, phy, rng, eve_density, guard_radius)
    G = cn((T, r.shape[1], M), rng)
    beta_e = pathloss_gain(r, phy)
    sig = np.abs(np.einsum("bnm,bm->bn", G.conj(), w)) ** 2
    if Z.shape[2] > 0 and phi_e < 1.0:
        an = (np.abs(np.einsum("bnm,bmk->bnk", G.conj(), Z)) ** 2).sum(axis=2)
        an_pow = (1 - phi_e) * P * beta_e * an / (M - K)
    else:
        an_pow = 0.0
    g_eve = np.where(mask, phi_e * P * beta_e * sig / (an_pow + N0), 0.0).max(axis=1)
    cs = np.maximum(np.log2(1 + g_sub.min(axis=1)) - np.log2(1 + g_eve), 0.0)
    return g_sub, cs
