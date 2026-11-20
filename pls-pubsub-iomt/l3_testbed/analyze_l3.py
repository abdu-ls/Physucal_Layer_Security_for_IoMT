"""Turn the raw publisher/subscriber logs of one testbed run into messages.csv and summary.csv.

messages.csv  one row per (message, intended subscriber):
  msg_id, topic, priority, sensitivity, publisher, t_pub, subscriber, t_recv, latency_s,
  tier, phi, mode, size_B, lost, miss, secrecy_outage
summary.csv   one row per topic, with message-level latency = time until ALL subscribers have it
              (the same definition as the L2 simulator).

Secrecy outage is joined from secrecy_traces.csv by topic, the decision (mode, phi) and the
trace step at publish time.

  python -m l3_testbed.analyze_l3 results/l3/<run>        (re-run analysis on an existing run)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_topics  # noqa: E402


def analyze(run, warmup=60.0, traces_dir=None):
    run = Path(run)
    logs = run / "shared" / "logs"
    pub = pd.concat([pd.read_csv(f) for f in sorted(logs.glob("pub_*.csv"))], ignore_index=True)
    sub_files = sorted(logs.glob("sub_*.csv"))
    sub = pd.concat([pd.read_csv(f) for f in sub_files], ignore_index=True) if sub_files else \
        pd.DataFrame(columns=["msg_id", "topic", "subscriber", "broker", "t_pub", "t_recv"])
    topics = {t["name"]: t for t in load_topics()["topics"]}
    t0 = pub.t_pub.min()
    pub = pub[pub.t_pub >= t0 + warmup].copy()

    # expected (message, subscriber) pairs
    exp = pub.assign(subscriber=pub.topic.map(lambda n: topics[n]["subscribers"])).explode("subscriber")
    first = sub.groupby(["msg_id", "subscriber"], as_index=False).t_recv.min()
    m = exp.merge(first, on=["msg_id", "subscriber"], how="left")
    m["latency_s"] = m.t_recv - m.t_pub
    m["lost"] = m.t_recv.isna().astype(int)
    m["priority"] = m.topic.map(lambda n: topics[n]["priority"])
    m["sensitivity"] = m.topic.map(lambda n: topics[n]["sensitivity"])
    m["deadline_s"] = m.topic.map(lambda n: topics[n]["deadline_s"])
    m["miss"] = ((m.lost == 1) | (m.latency_s > m.deadline_s)).astype(int)

    # secrecy outage from the L1 trace
    m["secrecy_outage"] = np.nan
    if traces_dir is not None and (Path(traces_dir) / "secrecy_traces.csv").exists():
        sec = pd.read_csv(Path(traces_dir) / "secrecy_traces.csv")
        steps = np.sort(sec.t.unique())
        step = steps[1] - steps[0]
        sec["k"] = np.round(sec.t / step).astype(int)
        sec["phi"] = sec.phi.round(3)
        m["k"] = (np.floor((m.t_pub - t0) / step).astype(int)) % len(steps)
        m["phi"] = m.phi.astype(float).round(3)
        m = m.drop(columns="secrecy_outage").merge(
            sec[["topic", "mode", "phi", "k", "secrecy_outage"]], on=["topic", "mode", "phi", "k"], how="left")
        m = m.drop(columns="k")
    cols = ["msg_id", "topic", "priority", "sensitivity", "publisher", "t_pub", "subscriber", "t_recv",
            "latency_s", "tier", "phi", "mode", "size_B", "lost", "miss", "secrecy_outage"]
    m[cols].to_csv(run / "messages.csv", index=False)

    # message-level summary (latency until all subscribers received it)
    g = m.groupby("msg_id").agg(topic=("topic", "first"), priority=("priority", "first"),
                                sensitivity=("sensitivity", "first"), latency_s=("latency_s", "max"),
                                lost=("lost", "max"), deadline_s=("deadline_s", "first"),
                                tier=("tier", "first"), mode=("mode", "first"), phi=("phi", "first"),
                                secrecy_outage=("secrecy_outage", "first"))
    g["miss"] = ((g.lost == 1) | (g.latency_s > g.deadline_s)).astype(int)
    s = g.groupby(["topic", "priority", "sensitivity"]).agg(
        n=("miss", "size"), mean_lat=("latency_s", "mean"), p95_lat=("latency_s", lambda x: x.quantile(0.95)),
        miss_rate=("miss", "mean"), lost_rate=("lost", "mean"), sec_out_rate=("secrecy_outage", "mean"),
        share_cloud=("tier", "mean"), share_split=("mode", lambda x: (x == "split").mean()),
        mean_phi=("phi", "mean")).reset_index()
    s.to_csv(run / "summary.csv", index=False)
    print(s.round(4).to_string(index=False))
    return s


if __name__ == "__main__":
    run = Path(sys.argv[1])
    import json
    meta = json.loads((run / "meta.json").read_text())
    tr = meta.get("traces")
    analyze(run, warmup=float(meta.get("warmup", 60)), traces_dir=Path(tr).parent if tr and tr != "None" else None)
