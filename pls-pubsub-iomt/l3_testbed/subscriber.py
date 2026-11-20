"""Subscriber container (nurse station, handheld, EHR gateway).

Connects to BOTH brokers and subscribes to every topic whose subscriber list contains this role,
so it receives a message whichever tier the orchestrator chose. Logs one line per message.

  python3 /repo/l3_testbed/subscriber.py --name nurse_station --edge 10.0.0.2 --cloud 10.0.0.3 --shared /shared
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time

import paho.mqtt.client as mqtt
import yaml


def make_client(cid):
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid)
    except AttributeError:
        return mqtt.Client(client_id=cid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--edge", default="10.0.0.2")
    ap.add_argument("--cloud", default="10.0.0.3")
    ap.add_argument("--shared", default="/shared")
    ap.add_argument("--topics", default="/repo/config/topics.yaml")
    ap.add_argument("--tls", action="store_true")
    ap.add_argument("--qos", type=int, default=0)
    a = ap.parse_args()

    with open(a.topics) as f:
        topics = [t for t in yaml.safe_load(f)["topics"] if a.name in t["subscribers"]]
    log_path = os.path.join(a.shared, "logs", f"sub_{a.name}.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "w", buffering=1 << 16)
    log.write("msg_id,topic,subscriber,broker,t_pub,t_recv\n")
    lock = threading.Lock()
    port = 8883 if a.tls else 1883

    def on_message(client, userdata, msg, *args):
        t_recv = time.time()
        try:
            h = json.loads(msg.payload.decode().rstrip())
        except ValueError:
            return
        with lock:
            log.write(f"{h['id']},{h['topic']},{a.name},{userdata},{h['t']:.6f},{t_recv:.6f}\n")

    def on_connect(client, userdata, flags, rc, *args):
        for t in topics:
            client.subscribe(t["mqtt"], qos=a.qos)

    clients = []
    for broker, host in (("edge", a.edge), ("cloud", a.cloud)):
        try:
            c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{a.name}-{broker}", userdata=broker)
        except AttributeError:
            c = mqtt.Client(client_id=f"{a.name}-{broker}", userdata=broker)
        if a.tls:
            c.tls_set(ca_certs=os.path.join(a.shared, "certs", "ca.crt"))
            c.tls_insecure_set(True)
        c.on_connect, c.on_message = on_connect, on_message
        c.connect(host, port, keepalive=60)
        c.loop_start()
        clients.append(c)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    while not stop.wait(1.0):
        with lock:
            log.flush()
    for c in clients:
        c.loop_stop()
        c.disconnect()
    with lock:
        log.close()


if __name__ == "__main__":
    main()
