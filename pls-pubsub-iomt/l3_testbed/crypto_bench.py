"""Cryptographic overhead benchmark for baseline B3 (runs inside a CPU-capped publisher container).

For each payload size in the topic set and for plain MQTT vs TLS 1.3 it measures:
  handshake_ms     TCP (+TLS) connect until CONNACK
  rtt_ms           QoS 1 publish until PUBACK (mean and 95th percentile)
  cpu_ms_per_msg   client CPU time per published message
and separately AES-128-GCM payload encryption time per operation.

  python3 /repo/l3_testbed/crypto_bench.py --edge 10.0.0.2 --shared /shared --n 500
"""
from __future__ import annotations

import argparse
import os
import statistics
import threading
import time

import paho.mqtt.client as mqtt
import yaml
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def make_client(cid):
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid)
    except AttributeError:
        return mqtt.Client(client_id=cid)


def bench_mqtt(host, tls, shared, size, n):
    c = make_client(f"bench-{'tls' if tls else 'plain'}-{size}")
    connected = threading.Event()
    acked = {}
    c.on_connect = lambda *a, **k: connected.set()

    def on_publish(client, userdata, mid, *args):
        acked[mid] = time.perf_counter()

    c.on_publish = on_publish
    if tls:
        c.tls_set(ca_certs=os.path.join(shared, "certs", "ca.crt"))
        c.tls_insecure_set(True)
    t0 = time.perf_counter()
    c.connect(host, 8883 if tls else 1883, keepalive=60)
    c.loop_start()
    if not connected.wait(10):
        raise RuntimeError("connect timeout")
    handshake = (time.perf_counter() - t0) * 1000
    payload = b"x" * size
    sent = {}
    cpu0 = time.process_time()
    for _ in range(n):
        ts = time.perf_counter()
        info = c.publish("bench/crypto", payload, qos=1)
        sent[info.mid] = ts
        info.wait_for_publish(timeout=5)
    cpu = (time.process_time() - cpu0) * 1000 / n
    rtts = [(acked[m] - sent[m]) * 1000 for m in sent if m in acked]
    c.loop_stop()
    c.disconnect()
    rtts.sort()
    return dict(handshake_ms=handshake, rtt_ms_mean=statistics.mean(rtts),
                rtt_ms_p95=rtts[int(0.95 * (len(rtts) - 1))], cpu_ms_per_msg=cpu, n_ok=len(rtts))


def bench_aes(size, n):
    key = AESGCM.generate_key(bit_length=128)
    aes = AESGCM(key)
    data = os.urandom(size)
    nonce = os.urandom(12)
    t0 = time.process_time()
    for _ in range(n):
        aes.encrypt(nonce, data, None)
    return (time.process_time() - t0) * 1e6 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default="10.0.0.2")
    ap.add_argument("--shared", default="/shared")
    ap.add_argument("--topics", default="/repo/config/topics.yaml")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="/shared/crypto.csv")
    a = ap.parse_args()
    with open(a.topics) as f:
        sizes = sorted({t["payload_bytes"] for t in yaml.safe_load(f)["topics"]})
    rows = ["mode,payload_bytes,rep,handshake_ms,rtt_ms_mean,rtt_ms_p95,cpu_ms_per_msg,aes_us_per_op,n_ok"]
    for rep in range(a.reps):
        for size in sizes:
            aes = bench_aes(size, 2000)
            for tls in (False, True):
                r = bench_mqtt(a.edge, tls, a.shared, size, a.n)
                rows.append(f"{'tls' if tls else 'plain'},{size},{rep},{r['handshake_ms']:.3f},{r['rtt_ms_mean']:.3f},"
                            f"{r['rtt_ms_p95']:.3f},{r['cpu_ms_per_msg']:.4f},{aes:.2f},{r['n_ok']}")
                print(rows[-1], flush=True)
    with open(a.out, "w") as f:
        f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
