"""Analytical / semi-analytical SOP for AN-aided multicast with PPP eavesdroppers.

Eavesdropper side (exact for any K, since w is unit-norm and orthogonal to the AN space Z):
    |g^H w|^2 ~ Exp(1),  ||g^H Z||^2 ~ Gamma(M-K, 1), independent.
    Pr(gamma_e(r) > x) = exp(-x / (phi * snr0(r))) * (1 + x(1-phi) / (phi (M-K)))^-(M-K)
Strongest non-colluding eavesdropper, PPP of density lambda in annulus [r0, R] (PGFL):
    F(x) = Pr(max_e gamma_e <= x) = exp(-2 pi lambda * int_{r0}^{R} Pr(gamma_e(r) > x) r dr)
Secrecy outage, R_s target, gamma_min = weakest subscriber SNR:
    SOP = 1 - E[ F((1+gamma_min) 2^-R_s - 1) ],   with F(x) = 0 for x < 0.

K = 1 (MRT):  gamma = phi * snr0(d) * Y,  Y ~ Gamma(M, 1)  -> fully analytical (1-D integral).
K > 1:        the law of gamma_min under the sum-MRT multicast beamformer has no simple form,
              so it is sampled; the eavesdropper part stays analytical  -> "semi-analytical".
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from common import snr0
from l1_phy.channel import beamformer_and_null, cn, effective_phi

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 1.x and 2.x


def _eve_cdf(x, phi, M, K, phy, eve_density=None, guard_radius=None, nr=400):
    """F(x) for an array x (any shape). x < 0 gives 0."""
    lam = phy["eve_density"] if eve_density is None else eve_density
    r0 = phy["guard_radius_m"] if guard_radius is None else guard_radius
    R = phy["eve_outer_radius_m"]
    phi_e = effective_phi(phi, M, K)
    x = np.asarray(x, dtype=float)
    xs = np.clip(x, 0.0, None)[..., None]                       # (..., 1)
    r = np.linspace(r0, R, nr)                                   # (nr,)
    s0 = snr0(r, phy)
    p = np.exp(-xs / (phi_e * s0))
    n_an = M - K
    if phi_e < 1.0 and n_an > 0:
        p = p * (1.0 + xs * (1 - phi_e) / (phi_e * n_an)) ** (-n_an)
    integral = _trapz(p * r, r, axis=-1)
    F = np.exp(-2 * np.pi * lam * integral)
    return np.where(x < 0, 0.0, F)


def sop_k1(phi, M, d, phy, eve_density=None, guard_radius=None, ny=3000):
    """Exact SOP and COP for K = 1 (single subscriber at distance d, MRT)."""
    phi_e = effective_phi(phi, M, 1)
    a = phi_e * snr0(d, phy)
    Rs, Rmin = phy["secrecy_rate_target"], phy["min_decode_rate"]
    # integrate over Y ~ Gamma(M,1) on a log grid
    y = np.geomspace(1e-8, stats.gamma.ppf(1 - 1e-12, M), ny)
    pdf = stats.gamma.pdf(y, M)
    x = (1 + a * y) * 2.0 ** (-Rs) - 1
    F = _eve_cdf(x, phi, M, 1, phy, eve_density, guard_radius)
    p_secure = _trapz(F * pdf, y) + 0.0
    sop = float(np.clip(1 - p_secure, 0, 1))
    cop = float(stats.gamma.cdf((2 ** Rmin - 1) / a, M))
    return {"sop": sop, "cop": cop}


def sop_semi(phi, M, K, sub_dist, phy, rng, n=20000, eve_density=None, guard_radius=None):
    """Semi-analytical SOP and COP for K >= 1 (subscriber channels sampled, eavesdroppers analytical)."""
    if K == 1 and np.ndim(sub_dist) == 0:
        return sop_k1(phi, M, float(sub_dist), phy, eve_density, guard_radius)
    sub_dist = np.broadcast_to(np.asarray(sub_dist, dtype=float), (K,))
    phi_e = effective_phi(phi, M, K)
    H = cn((n, K, M), rng)
    w, _ = beamformer_and_null(H)
    gain = np.abs(np.einsum("bkm,bm->bk", H.conj(), w)) ** 2
    gmin = (phi_e * snr0(sub_dist, phy)[None, :] * gain).min(axis=1)
    Rs, Rmin = phy["secrecy_rate_target"], phy["min_decode_rate"]
    x = (1 + gmin) * 2.0 ** (-Rs) - 1
    F = _eve_cdf(x, phi, M, K, phy, eve_density, guard_radius)
    return {"sop": float(1 - F.mean()), "cop": float((np.log2(1 + gmin) < Rmin).mean())}
