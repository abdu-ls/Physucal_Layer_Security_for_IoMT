"""Containernet experiment: one scheme, one repetition (run with root, from the Containernet venv).

Topology (all hosts are Docker containers from the iomt-node image):

  p1..p6 (publishers, CPU-capped) --3 ms, 2 Mbit/s-- s1 (AP switch) --- edge broker 10.0.0.2
  u1..u4 (subscribers) ------- trace-shaped netem --- s1
  s1 ----- 30 ms backhaul ----- s2 --- cloud broker 10.0.0.3

Examples (repo root, after setup_host.sh):
  sudo -E ~/cn-venv/bin/python l3_testbed/run_experiment.py --scheme proposed \
       --policy results/l2/train_proposed_<stamp>_s0/policy.npz --rep 0
  sudo -E ~/cn-venv/bin/python l3_testbed/run_experiment.py --scheme B4_greedy --policy greedy --rep 0
  sudo -E ~/cn-venv/bin/python l3_testbed/run_experiment.py --scheme B3_crypto --policy crypto --tls --rep 0
  sudo -E ~/cn-venv/bin/python l3_testbed/run_experiment.py --crypto-bench
  add --duration 60 --warmup 10 for a short smoke run; --controller none to run without Ryu.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mininet.link import TCLink                                      # noqa: E402
from mininet.log import info, setLogLevel                            # noqa: E402
from mininet.net import Containernet                                 # noqa: E402
from mininet.node import OVSSwitch, RemoteController                 # noqa: E402

from common import RESULTS_DIR, load_params, load_topics, write_meta  # noqa: E402

BROKER_CONF = """per_listener_settings false
allow_anonymous true
persistence false
log_dest file /shared/logs/{name}_broker.log
max_queued_messages 100000
listener 1883 0.0.0.0
"""
TLS_CONF = """listener 8883 0.0.0.0
cafile /shared/certs/ca.crt
certfile /shared/certs/server.crt
keyfile /shared/certs/server.key
tls_version tlsv1.3
"""


def latest_traces():
    cands = sorted((RESULTS_DIR / "l1").glob("l1_*/link_traces.csv"))
    if not cands:
        sys.exit("No link_traces.csv found: run `python -m l1_phy.run_l1 --part traces` and copy results/l1 here.")
    return cands[-1]


def build(args, shared, n_pub, cpu_pub, edge_cpu, bench=False):
    params = load_params()
    roles = list(load_topics()["testbed"]["subscribers"])
    if args.controller == "ryu":
        net = Containernet(controller=RemoteController, switch=OVSSwitch)
        net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6653)
        sw_kw = {}
    else:
        net = Containernet(controller=None, switch=OVSSwitch)
        sw_kw = {"failMode": "standalone"}
    vols = [f"{REPO}:/repo:ro", f"{shared}:/shared:rw"]
    img = args.image
    s1 = net.addSwitch("s1", **sw_kw)
    s2 = net.addSwitch("s2", **sw_kw)
    edge = net.addDocker("edge", ip="10.0.0.2", dimage=img, volumes=vols,
                         cpu_period=100000, cpu_quota=int(edge_cpu * 100000))
    cloud = net.addDocker("cloud", ip="10.0.0.3", dimage=img, volumes=vols)
    net.addLink(edge, s1)
    bh = f"{params['net']['backhaul_delay_s'] * 1000:.0f}ms"
    net.addLink(s1, s2, cls=TCLink, delay=bh)
    net.addLink(cloud, s2)
    pubs, subs, sub_intf = [], [], {}
    for i in range(n_pub):
        p = net.addDocker(f"p{i + 1}", ip=f"10.0.0.{21 + i}", dimage=img, volumes=vols,
                          cpu_period=100000, cpu_quota=int(cpu_pub * 100000))
        net.addLink(p, s1, cls=TCLink, delay="3ms", jitter="1ms", bw=2)
        pubs.append(p)
    if not bench:
        for i, role in enumerate(roles):
            u = net.addDocker(f"u{i + 1}", ip=f"10.0.0.{11 + i}", dimage=img, volumes=vols)
            link = net.addLink(u, s1)
            sub_intf[role] = link.intf2.name           # switch-side interface, host namespace
            subs.append((role, u))
    return net, edge, cloud, pubs, subs, sub_intf


def start_brokers(edge, cloud, shared, tls):
    for name, node in (("edge", edge), ("cloud", cloud)):
        conf = BROKER_CONF.format(name=name) + (TLS_CONF if tls else "")
        (shared / f"{name}.conf").write_text(conf)
        node.cmd(f"mosquitto -c /shared/{name}.conf -d")
    time.sleep(2)


def chown(path):
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid and gid:
        os.system(f"chown -R {uid}:{gid} '{path}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="proposed")
    ap.add_argument("--policy", default="greedy", help="policy.npz path or greedy|fixed_an|crypto|random|no_an")
    ap.add_argument("--variant", default=None, help="env variant the policy was trained with (default from scheme)")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--warmup", type=float, default=60)
    ap.add_argument("--tls", action="store_true", help="publishers and subscribers use TLS 1.3 (B3)")
    ap.add_argument("--controller", default="ryu", choices=["ryu", "none"])
    ap.add_argument("--image", default="iomt-node")
    ap.add_argument("--edge-cpu", type=float, default=1.0, help="edge broker CPU share (cores); calibrate to edge_service_rate")
    ap.add_argument("--pub-cpu", type=float, default=0.10, help="CPU share per publisher container")
    ap.add_argument("--traces", default=None, help="link_traces.csv (default: latest L1 run)")
    ap.add_argument("--load", type=float, default=1.0)
    ap.add_argument("--crypto-bench", action="store_true", help="run the crypto overhead benchmark instead")
    ap.add_argument("--cli", action="store_true", help="drop into the Containernet CLI before stopping")
    a = ap.parse_args()
    setLogLevel("info")

    topics = load_topics()
    tb = topics["testbed"]
    tag = "crypto_bench" if a.crypto_bench else a.scheme
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run = RESULTS_DIR / "l3" / f"{tag}_{stamp}_r{a.rep}"
    shared = run / "shared"
    (shared / "logs").mkdir(parents=True, exist_ok=True)
    certs = RESULTS_DIR / "l3" / "certs"
    if certs.exists():
        shutil.copytree(certs, shared / "certs")
    elif a.tls or a.crypto_bench:
        sys.exit("TLS needs certificates: run `bash l3_testbed/gen_certs.sh results/l3` first.")
    traces = Path(a.traces) if a.traces else (None if a.crypto_bench else latest_traces())
    write_meta(run, scheme=a.scheme, policy=a.policy, rep=a.rep, duration=a.duration, warmup=a.warmup,
               tls=a.tls, controller=a.controller, edge_cpu=a.edge_cpu, pub_cpu=a.pub_cpu,
               traces=str(traces), load=a.load)

    n_pub = 1 if a.crypto_bench else tb["publisher_containers"]
    net, edge, cloud, pubs, subs, sub_intf = build(a, shared, n_pub, a.pub_cpu, a.edge_cpu, bench=a.crypto_bench)
    orch = rep = None
    try:
        net.start()
        start_brokers(edge, cloud, shared, tls=a.tls or a.crypto_bench)
        if a.crypto_bench:
            info("*** Crypto benchmark (several minutes)\n")
            print(pubs[0].cmd("python3 /repo/l3_testbed/crypto_bench.py --edge 10.0.0.2 --shared /shared "
                              "--out /shared/crypto.csv"))
            shutil.copy(shared / "crypto.csv", run / "crypto.csv")
            return

        from l3_testbed.orchestrator import Orchestrator
        from l3_testbed.replay_traces import TraceReplayer
        variant = a.variant or {"B1_agnostic": "agnostic", "A_no_priority": "no_priority",
                                "A_no_sensitivity": "no_sensitivity", "A_no_sdn": "no_sdn",
                                "A_no_an": "no_an"}.get(a.scheme, "proposed")
        orch = Orchestrator(a.policy, shared, variant=variant, load=a.load, seed=a.rep, crypto=a.tls)
        orch.decide_epoch()                     # warm the twin
        orch.start()
        rep = TraceReplayer(traces, topics, sub_intf, orch, step_s=tb["trace_step_s"])
        rep.start()
        tls = "--tls" if a.tls else ""
        for role, u in subs:
            u.cmd(f"nohup python3 /repo/l3_testbed/subscriber.py --name {role} --shared /shared {tls} "
                  f"> /shared/logs/{role}.out 2>&1 &")
        time.sleep(2)
        total = a.warmup + a.duration
        for i, p in enumerate(pubs):
            p.cmd(f"nohup python3 /repo/l3_testbed/publisher.py --name {p.name} --devices {tb['devices_per_container']} "
                  f"--shared /shared --duration {total} --load {a.load} --seed {a.rep * 100 + i} {tls} "
                  f"> /shared/logs/{p.name}.out 2>&1 &")
        info(f"*** Running {a.scheme}: {a.warmup:.0f}s warm-up + {a.duration:.0f}s measurement\n")
        t_end = time.time() + total + 5
        while time.time() < t_end:
            time.sleep(min(30, max(0.0, t_end - time.time())))
            info(f"    {max(0, t_end - time.time()):.0f}s left, netem updates so far: {rep.applied}\n")
        if a.cli:
            from mininet.cli import CLI
            CLI(net)
        for _, u in subs:
            u.cmd("pkill -INT -f subscriber.py")
        time.sleep(3)
    finally:
        if orch:
            orch.stop()
        if rep:
            rep.stop()
        net.stop()
        chown(run)
    if not a.crypto_bench:
        from l3_testbed.analyze_l3 import analyze
        analyze(run, warmup=a.warmup, traces_dir=traces.parent)
        chown(run)
    print(f"Results in {run}")


if __name__ == "__main__":
    main()
