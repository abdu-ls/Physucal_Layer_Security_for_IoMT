"""IoMT publisher container: emulates several devices, each publishing every topic as a Poisson process.

For each message it reads the orchestrator's current decision for that topic (decisions.json in
the shared folder), publishes to the edge or cloud broker accordingly, and logs what it sent.
Payload: JSON header (msg_id, topic, t_pub, tier, phi, mode) padded to the topic's payload size.

Run inside a container (run_experiment.py does this):
  python3 /repo/l3_testbed/publisher.py --name pub1 --devices 5 --edge 10.0.0.2 --cloud 10.0.0.3 \
      --shared /shared --duration 660 [--tls] [--qos 0]
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import random
import signal
import time

import paho.mqtt.client as mqtt
import yaml


def make_client(cid):
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid)
    except AttributeError:          # paho-mqtt 1.x
        return mqtt.Client(client_id=cid)


def connect(host, port, cid, tls, shared):
    c = make_client(cid)
    if tls:
        c.tls_set(ca_certs=os.path.join(shared, "certs", "ca.crt"))
        c.tls_insecure_set(True)
    c.max_inflight_messages_set(1000)
    c.max_queued_messages_set(0)
    c.connect(host, port, keepalive=60)
    c.loop_start()
    return c


class Decisions:
    """Reads decisions.json when it changes (written atomically by the orchestrator)."""

    def __init__(self, path):
        self.path, self.mtime, self.d = path, 0.0, {}

    def get(self, topic):
        try:
            m = os.path.getmtime(self.path)
            if m != self.mtime:
                with open(self.path) as f:
                    self.d = json.load(f)
                self.mtime = m
        except (OSError, ValueError):
            pass
        return self.d.get(topic, {"tier": 0, "mode": "single", "phi": 1.0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--devices", type=int, default=5)
    ap.add_argument("--edge", default="10.0.0.2")
    ap.add_argument("--cloud", default="10.0.0.3")
    ap.add_argument("--shared", default="/shared")
    ap.add_argument("--topics", default="/repo/config/topics.yaml")
    ap.add_argument("--duration", type=float, default=660)
    ap.add_argument("--load", type=float, default=1.0)
    ap.add_argument("--tls", action="store_true")
    ap.add_argument("--qos", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    with open(a.topics) as f:
        topics = yaml.safe_load(f)["topics"]
    rnd = random.Random(hash((a.name, a.seed)) & 0xFFFFFFFF)
    port = 8883 if a.tls else 1883
    clients = {0: connect(a.edge, port, f"{a.name}-edge", a.tls, a.shared),
               1: connect(a.cloud, port, f"{a.name}-cloud", a.tls, a.shared)}
    dec = Decisions(os.path.join(a.shared, "decisions.json"))
    log_path = os.path.join(a.shared, "logs", f"pub_{a.name}.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "w", buffering=1 << 16)
    log.write("msg_id,topic,publisher,t_pub,tier,phi,mode,size_B\n")

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    t0 = time.time()
    heap = []
    for dv in range(a.devices):
        for tp in topics:
            rate = tp["rate_per_device"] * a.load
            heapq.heappush(heap, (t0 + rnd.expovariate(rate), dv, tp["name"]))
    by_name = {tp["name"]: tp for tp in topics}
    seq = 0
    while heap and not stop["flag"]:
        t_next, dv, name = heapq.heappop(heap)
        if t_next - t0 > a.duration:
            break
        delay = t_next - time.time()
        if delay > 0:
            time.sleep(delay)
        tp = by_name[name]
        d = dec.get(name)
        seq += 1
        msg_id = f"{a.name}-{seq}"
        t_pub = time.time()
        head = json.dumps({"id": msg_id, "topic": name, "t": t_pub, "tier": d["tier"], "phi": d["phi"],
                           "mode": d["mode"], "dev": dv})
        payload = head.ljust(max(tp["payload_bytes"], len(head)))
        clients[int(d["tier"])].publish(tp["mqtt"], payload, qos=a.qos)
        log.write(f"{msg_id},{name},{a.name},{t_pub:.6f},{d['tier']},{d['phi']},{d['mode']},{len(payload)}\n")
        heapq.heappush(heap, (t_next + rnd.expovariate(tp["rate_per_device"] * a.load), dv, name))
    time.sleep(1.0)
    for c in clients.values():
        c.loop_stop()
        c.disconnect()
    log.close()


if __name__ == "__main__":
    main()
