"""Testbed smoke test (environment-setup steps 6 and 7).

Builds 1 switch + controller + an MQTT broker + 2 clients, checks pub/sub across the switch,
then applies a netem delay on one link and checks that the round-trip time changes.

  sudo -E ~/cn-venv/bin/python l3_testbed/smoke_test.py               # with Ryu (start it first)
  sudo -E ~/cn-venv/bin/python l3_testbed/smoke_test.py --controller none
"""
import argparse
import re
import subprocess
import time

from mininet.log import setLogLevel
from mininet.net import Containernet
from mininet.node import OVSSwitch, RemoteController


def rtt(node, ip):
    out = node.cmd(f"ping -c 5 -i 0.2 {ip}")
    m = re.search(r"= [\d.]+/([\d.]+)/", out)
    return float(m.group(1)) if m else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="ryu", choices=["ryu", "none"])
    ap.add_argument("--image", default="iomt-node")
    a = ap.parse_args()
    setLogLevel("info")
    if a.controller == "ryu":
        net = Containernet(controller=RemoteController, switch=OVSSwitch)
        net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6653)
        kw = {}
    else:
        net = Containernet(controller=None, switch=OVSSwitch)
        kw = {"failMode": "standalone"}
    s1 = net.addSwitch("s1", **kw)
    b = net.addDocker("b1", ip="10.0.0.2", dimage=a.image)
    c1 = net.addDocker("c1", ip="10.0.0.11", dimage=a.image)
    c2 = net.addDocker("c2", ip="10.0.0.12", dimage=a.image)
    net.addLink(b, s1)
    net.addLink(c1, s1)
    l2 = net.addLink(c2, s1)
    ok = True
    try:
        net.start()
        b.cmd("printf 'listener 1883 0.0.0.0\\nallow_anonymous true\\n' > /tmp/m.conf && mosquitto -c /tmp/m.conf -d")
        time.sleep(2)
        c2.cmd("mosquitto_sub -h 10.0.0.2 -t smoke/test -C 1 -W 10 > /tmp/got.txt &")
        time.sleep(1)
        c1.cmd("mosquitto_pub -h 10.0.0.2 -t smoke/test -m hello-iomt")
        time.sleep(2)
        got = c2.cmd("cat /tmp/got.txt").strip()
        print(f"\n[check] pub/sub across s1: {'PASS' if 'hello-iomt' in got else 'FAIL'} (received: {got!r})")
        ok &= "hello-iomt" in got
        r0 = rtt(c2, "10.0.0.2")
        intf = l2.intf2.name
        subprocess.run(f"tc qdisc replace dev {intf} root netem delay 50ms", shell=True, check=True)
        r1 = rtt(c2, "10.0.0.2")
        passed = r1 - r0 > 40
        print(f"[check] netem on {intf}: RTT {r0:.2f} ms -> {r1:.2f} ms : {'PASS' if passed else 'FAIL'}")
        ok &= passed
        lim = c1.cmd("cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us").strip()
        print(f"[info] container CPU limit file reads: {lim!r}")
    finally:
        net.stop()
    print("\nSMOKE TEST", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    main()
