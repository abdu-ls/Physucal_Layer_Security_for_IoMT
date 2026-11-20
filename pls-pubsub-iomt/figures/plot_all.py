"""Make every paper figure (F1-F12) from the latest results. Missing inputs are skipped.

  python figures/plot_all.py            # writes figures/F*.pdf and .png
  python figures/plot_all.py --only F1 F6

Figures are sized for an IEEE single column (3.5 in). Zero probabilities are not drawn on log axes. Series colours follow the scheme, never
its rank, and lines/bars also carry markers or hatches so they read in greyscale print.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS_DIR, load_params  # noqa: E402

OUT = Path(__file__).resolve().parent
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]
HATCH = ["", "//", "\\\\", "xx", "..", "++", "--", "oo"]
SCHEMES = ["proposed", "B1_agnostic", "B2_fixed_an", "B3_crypto", "B4_greedy", "B5_random"]
LABEL = {"proposed": "Proposed", "B1_agnostic": "B1 agnostic DQN", "B2_fixed_an": "B2 fixed AN",
         "B3_crypto": "B3 crypto-only", "B4_greedy": "B4 greedy", "B5_random": "B5 random",
         "A_no_priority": "w/o priority", "A_no_sensitivity": "w/o sensitivity", "A_no_sdn": "w/o SDN view",
         "A_no_an": "w/o AN"}
COLOR = {s: PAL[i] for i, s in enumerate(SCHEMES)}
COLOR.update({"A_no_priority": PAL[6], "A_no_sensitivity": PAL[7], "A_no_sdn": PAL[3], "A_no_an": PAL[1]})

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e4df", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8980",
    "lines.linewidth": 1.6, "lines.markersize": 4.5, "legend.frameon": False, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
W1 = (3.5, 2.5)
W2 = (7.16, 2.5)


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def latest_file(layer, pattern):
    c = sorted((RESULTS_DIR / layer).glob(pattern))
    return c[-1] if c else None


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return 0.0 if len(x) < 2 else 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def weighted(df, keys, col):
    """Message-weighted mean of col per keys + seed, then mean and 95% CI across seeds."""
    g = df.assign(_w=df[col] * df.n).groupby(keys + ["seed"]).agg(_w=("_w", "sum"), n=("n", "sum"))
    g["v"] = g._w / g.n
    out = g.groupby(keys).v.agg(["mean", ci95]).reset_index()
    return out.rename(columns={"ci95": "ci"})


# ---------------------------------------------------------------- L1
def f1(sop):
    d = sop[sop.exp == "F1"]
    fig, ax = plt.subplots(figsize=W1)
    for i, M in enumerate(sorted(d.M.unique())):
        a = d[(d.M == M) & (d.method == "analytic")].sort_values("phi")
        m = d[(d.M == M) & (d.method == "mc")].sort_values("phi")
        ax.plot(a.phi, a.sop.where(lambda v: v > 0), color=PAL[i], label=f"M = {M}, analysis")
        ax.plot(m.phi, m.sop.where(lambda v: v > 0), MARK[i], color=PAL[i], mfc="none", label=f"M = {M}, Monte Carlo")
    ax.set_yscale("log")
    ax.set_xlabel("Signal power fraction φ  (1 = no AN)")
    ax.set_ylabel("Secrecy outage probability")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "F1_sop_vs_phi")


def f2(sop):
    d = sop[sop.exp == "F2"]
    fig, ax = plt.subplots(figsize=W1)
    for i, phi in enumerate(sorted(d.phi.unique())):
        a = d[(d.phi == phi) & (d.method != "mc")].sort_values("K")
        m = d[(d.phi == phi) & (d.method == "mc")].sort_values("K")
        lab = "no AN" if phi == 1.0 else f"φ = {phi:g}"
        ax.plot(a.K, a.sop.where(lambda v: v > 0), color=PAL[i], label=f"{lab}, analysis")
        ax.plot(m.K, m.sop.where(lambda v: v > 0), MARK[i], color=PAL[i], mfc="none", label=f"{lab}, Monte Carlo")
    ax.set_yscale("log")
    ax.set_xlabel("Subscribers per topic K  (M = 8)")
    ax.set_ylabel("Secrecy outage probability")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "F2_sop_vs_K")


def f3(sop):
    d = sop[sop.exp == "F3"]
    fig, axes = plt.subplots(1, 2, figsize=W2, sharey=True)
    for ax, phi in zip(axes, (0.5, 1.0)):
        for i, r0 in enumerate(sorted(d.guard_radius.unique())):
            a = d[(d.phi == phi) & (d.guard_radius == r0) & (d.method != "mc")].sort_values("eve_density")
            m = d[(d.phi == phi) & (d.guard_radius == r0) & (d.method == "mc")].sort_values("eve_density")
            ax.plot(a.eve_density, a.sop.where(lambda v: v > 0), color=PAL[i], label=f"r₀ = {r0:g} m")
            ax.plot(m.eve_density, m.sop.where(lambda v: v > 0), MARK[i], color=PAL[i], mfc="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Eavesdropper density λₑ (m⁻²)")
        ax.set_title("AN, φ = 0.5" if phi < 1 else "No AN")
    axes[0].set_ylabel("Secrecy outage probability")
    axes[1].legend(title="lines: analysis, markers: MC", loc="lower right")
    save(fig, "F3_sop_vs_density")


def f4(phi):
    fig, ax = plt.subplots(figsize=W1)
    for i, M in enumerate(sorted(phi.M.unique())):
        d = phi[phi.M == M].sort_values("K")
        ax.plot(d.K, d.phi_star_cop_capped, marker=MARK[i], color=PAL[i], label=f"M = {M}")
    ax.set_xlabel("Subscribers in the group K")
    ax.set_ylabel("Optimal φ* (COP ≤ 0.1)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    save(fig, "F4_phi_opt")


# ---------------------------------------------------------------- L2
def f5():
    fig, ax = plt.subplots(figsize=W1)
    found = False
    for i, (v, lab) in enumerate((("proposed", "Proposed"), ("agnostic", "B1 agnostic DQN"))):
        runs = [p for p in (RESULTS_DIR / "l2").glob(f"train_{v}_2*") if (p / "train_log.csv").exists()]
        if not runs:
            continue
        lens = {p: len(pd.read_csv(p / "train_log.csv")) for p in runs}
        full = max(lens.values())
        runs = [p for p in runs if lens[p] == full]          # drop smoke-test runs
        by_seed = {}
        for p in sorted(runs):                                # latest run per seed
            by_seed[p.name.rsplit("_s", 1)[1]] = p
        runs = list(by_seed.values())
        found = True
        logs = [pd.read_csv(p / "train_log.csv").ret.rolling(10, min_periods=1).mean().values for p in runs]
        n = min(len(l) for l in logs)
        arr = np.stack([l[:n] for l in logs])
        x = np.arange(1, n + 1)
        c = COLOR["proposed" if v == "proposed" else "B1_agnostic"]
        ax.plot(x, arr.mean(0), color=c, label=f"{lab} ({len(runs)} seeds)")
        if len(runs) > 1:
            ax.fill_between(x, arr.min(0), arr.max(0), color=c, alpha=0.15, lw=0)
    if not found:
        plt.close(fig)
        print("  skip F5: no train_log.csv")
        return
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return (10-ep. mean)")
    ax.set_yscale("symlog", linthresh=100)
    ax.legend(loc="lower right")
    save(fig, "F5_convergence")


def grouped_bars(ax, tab, groups, schemes, ylabel):
    w = 0.8 / len(schemes)
    x = np.arange(len(groups))
    for i, s in enumerate(schemes):
        d = tab[tab.scheme == s].set_index(tab.columns[1]).reindex(groups)
        ax.bar(x + (i - (len(schemes) - 1) / 2) * w, d["mean"], w * 0.92, yerr=d["ci"], color=COLOR[s],
               hatch=HATCH[i % len(HATCH)], edgecolor="white", linewidth=0.4, error_kw=dict(lw=0.6, capsize=1.5),
               label=LABEL[s])
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)


def f6_f7_f8_f10(ev, lam0):
    base = ev[(ev.load == 1.0) & (ev.eve_density == lam0)]
    schemes = [s for s in SCHEMES if s in base.scheme.unique()]
    # F6 deadline satisfaction per priority
    t = weighted(base, ["scheme", "priority"], "miss_rate")
    t["mean"] = 1 - t["mean"]
    fig, ax = plt.subplots(figsize=W2)
    grouped_bars(ax, t, ["P1", "P2", "P3"], schemes, "Deadline satisfaction")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=len(schemes), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "F6_deadline_by_priority")
    # F7 SOP violation per sensitivity (crypto-only excluded: confidentiality is computational there)
    s7 = [s for s in schemes if s != "B3_crypto"]
    t = weighted(base[base.scheme.isin(s7)], ["scheme", "sensitivity"], "viol_rate")
    fig, ax = plt.subplots(figsize=W2)
    grouped_bars(ax, t, ["High", "Medium", "Low"], s7, "Share of messages over SOP target")
    ax.legend(ncol=len(s7), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "F7_sop_violation_by_sensitivity")
    # F8 latency and energy vs load (two panels, one axis each)
    d = ev[ev.eve_density == lam0]
    fig, axes = plt.subplots(1, 2, figsize=W2)
    for col, ax, lab, k in (("mean_lat", axes[0], "Mean latency (ms)", 1000), ("energy_per_msg", axes[1],
                                                                                 "Energy per message (mJ)", 1000)):
        t = weighted(d, ["scheme", "load"], col)
        for i, s in enumerate(schemes):
            q = t[t.scheme == s].sort_values("load")
            ax.errorbar(q.load, q["mean"] * k, yerr=q.ci * k, marker=MARK[i], color=COLOR[s], label=LABEL[s],
                        capsize=1.5, elinewidth=0.6)
        ax.set_xlabel("Offered load (× nominal)")
        ax.set_ylabel(lab)
    axes[0].legend(ncol=2)
    save(fig, "F8_latency_energy_vs_load")
    # F10 ablations: P1 deadline satisfaction and High-sensitivity SOP violation
    abl = ["proposed", "A_no_priority", "A_no_sensitivity", "A_no_sdn", "A_no_an"]
    abl = [s for s in abl if s in base.scheme.unique()]
    if len(abl) > 1:
        fig, axes = plt.subplots(1, 2, figsize=W2)
        t1 = weighted(base[base.priority == "P1"], ["scheme"], "miss_rate").set_index("scheme").reindex(abl)
        t2 = weighted(base[base.sensitivity == "High"], ["scheme"], "viol_rate").set_index("scheme").reindex(abl)
        for ax, t, lab, inv in ((axes[0], t1, "P1 deadline satisfaction", True),
                                (axes[1], t2, "High-sensitivity SOP violation", False)):
            vals = 1 - t["mean"] if inv else t["mean"]
            ax.barh(range(len(abl)), vals, xerr=t.ci, color=[COLOR[s] for s in abl], edgecolor="white",
                    hatch=None, error_kw=dict(lw=0.6, capsize=1.5))
            ax.set_yticks(range(len(abl)))
            ax.set_yticklabels([LABEL[s] for s in abl])
            ax.invert_yaxis()
            ax.set_xlabel(lab)
            for j, v in enumerate(vals):
                ax.text(v, j, f" {v:.3f}", va="center", fontsize=6.5, color="#3d3d3a")
        save(fig, "F10_ablations")


def f9(sw):
    t = sw.assign(_l=sw.mean_lat * sw.n, _v=sw.viol_rate * sw.n).groupby(["beta_s", "seed"]).agg(
        l=("_l", "sum"), v=("_v", "sum"), n=("n", "sum"))
    t["lat"], t["viol"] = t.l / t.n * 1000, t.v / t.n
    g = t.groupby("beta_s").agg(lat=("lat", "mean"), viol=("viol", "mean")).reset_index().sort_values("beta_s")
    fig, ax = plt.subplots(figsize=W1)
    ax.plot(g.lat, g.viol, "-o", color=COLOR["proposed"])
    for _, r in g.iterrows():
        ax.annotate(f"β_s = {r.beta_s:g}", (r.lat, r.viol), textcoords="offset points", xytext=(4, 3), fontsize=6.5)
    ax.set_xlabel("Mean latency (ms)")
    ax.set_ylabel("Share of messages over SOP target")
    save(fig, "F9_secrecy_latency_tradeoff")


# ---------------------------------------------------------------- L3
def f11(ev, lam0):
    runs = [p for p in (RESULTS_DIR / "l3").glob("*_r*") if (p / "summary.csv").exists() and "crypto_bench" not in p.name]
    if not runs or ev is None:
        print("  skip F11: no L3 summaries or no L2 eval")
        return
    meas = pd.concat([pd.read_csv(p / "summary.csv").assign(scheme=p.name.split("_20")[0]) for p in runs])
    meas = meas.groupby(["scheme", "topic"]).mean_lat.mean().reset_index()
    sim = ev[(ev.load == 1.0) & (ev.eve_density == lam0)].groupby(["scheme", "topic"]).mean_lat.mean().reset_index()
    m = meas.merge(sim, on=["scheme", "topic"], suffixes=("_tb", "_sim"))
    fig, ax = plt.subplots(figsize=W1)
    for i, s in enumerate([s for s in SCHEMES if s in m.scheme.unique()]):
        d = m[m.scheme == s]
        ax.plot(d.mean_lat_sim * 1000, d.mean_lat_tb * 1000, MARK[i], color=COLOR[s], label=LABEL[s])
    lim = [0, max(m.mean_lat_sim.max(), m.mean_lat_tb.max()) * 1100]
    ax.plot(lim, lim, color="#8a8980", lw=0.8, ls="--")
    ax.set_xlabel("Simulated mean latency (ms)")
    ax.set_ylabel("Testbed mean latency (ms)")
    ax.legend()
    save(fig, "F11_testbed_vs_sim")


def f12():
    c = latest_file("l3", "crypto_bench_*/crypto.csv")
    if c is None:
        print("  skip F12: no crypto.csv")
        return
    d = pd.read_csv(c)
    fig, axes = plt.subplots(1, 2, figsize=W2)
    for i, mode in enumerate(("plain", "tls")):
        q = d[d["mode"] == mode].groupby("payload_bytes").agg(rtt=("rtt_ms_mean", "mean"), rci=("rtt_ms_mean", ci95),
                                                              cpu=("cpu_ms_per_msg", "mean"), cci=("cpu_ms_per_msg", ci95))
        lab = "Plain MQTT (AN-only delivery)" if mode == "plain" else "MQTT over TLS 1.3"
        axes[0].errorbar(q.index, q.rtt, yerr=q.rci, marker=MARK[i], color=PAL[i], label=lab, capsize=1.5)
        axes[1].errorbar(q.index, q.cpu, yerr=q.cci, marker=MARK[i], color=PAL[i], label=lab, capsize=1.5)
    for ax, lab in zip(axes, ("Publish-to-PUBACK time (ms)", "Client CPU time per message (ms)")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Payload (bytes)")
        ax.set_ylabel(lab)
    axes[0].legend()
    save(fig, "F12_crypto_overhead")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    want = lambda f: a.only is None or f in a.only
    lam0 = load_params()["phy"]["eve_density"]
    sop_p = latest_file("l1", "l1_*/sop_mc.csv")
    if sop_p is not None:
        sop = pd.read_csv(sop_p)
        if want("F1"): f1(sop)
        if want("F2"): f2(sop)
        if want("F3"): f3(sop)
    else:
        print("  skip F1-F3: no sop_mc.csv")
    phi_p = latest_file("l1", "l1_*/phi_opt.csv")
    if phi_p is not None and want("F4"):
        f4(pd.read_csv(phi_p))
    if want("F5"):
        f5()
    ev_p = latest_file("l2", "eval_*/eval.csv")
    ev = pd.read_csv(ev_p) if ev_p is not None else None
    if ev is not None and any(want(f) for f in ("F6", "F7", "F8", "F10")):
        f6_f7_f8_f10(ev, lam0)
    sw_p = latest_file("l2", "eval_*/sweep_beta.csv")
    if sw_p is not None and want("F9"):
        f9(pd.read_csv(sw_p))
    if want("F11"):
        f11(ev, lam0)
    if want("F12"):
        f12()


if __name__ == "__main__":
    main()
