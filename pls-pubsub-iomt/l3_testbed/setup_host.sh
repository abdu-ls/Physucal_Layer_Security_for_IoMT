#!/usr/bin/env bash
# One-time setup of the Containernet testbed on Ubuntu 24.04 (WSL2 or a VM).
# Run from the repo root:  bash l3_testbed/setup_host.sh
# Each step stops on error so you can see exactly where it failed. Re-running is safe.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$HOME/cn-venv"
CN="$HOME/containernet"

step() { echo; echo "=== $* ==="; }

step "1. WSL / systemd check"
if grep -qi microsoft /proc/version; then
  echo "WSL detected."
  if [ "$(ps -p 1 -o comm=)" != "systemd" ]; then
    echo "systemd is NOT running. Add the following to /etc/wsl.conf, then run 'wsl --shutdown' in PowerShell and reopen Ubuntu:"
    echo "  [boot]"
    echo "  systemd=true"
    exit 1
  fi
fi

step "2. Base packages"
sudo apt-get update
sudo apt-get install -y git ansible python3-venv python3-pip openvswitch-switch iproute2 util-linux openssl curl

step "3. Open vSwitch kernel module"
if ! sudo modprobe openvswitch; then
  echo "modprobe openvswitch FAILED: this kernel has no OVS module."
  echo "On WSL2 you need a custom kernel with CONFIG_OPENVSWITCH, or use the Ubuntu 24.04 VM fallback."
  exit 1
fi
lsmod | grep -q openvswitch && echo "openvswitch module loaded."
sudo systemctl enable --now openvswitch-switch
sudo ovs-vsctl show

step "4. Docker Engine (not Docker Desktop)"
if ! command -v docker >/dev/null || docker info 2>/dev/null | grep -qi "docker desktop"; then
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo systemctl enable --now docker
sudo docker run --rm hello-world

step "5. Containernet (venv install, required on 24.04)"
[ -d "$CN" ] || git clone https://github.com/containernet/containernet.git "$CN"
sudo ansible-playbook -i "localhost," -c local "$CN/ansible/install.yml"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install "$CN"
"$VENV/bin/pip" install numpy pandas pyyaml scipy

step "6. Images"
sudo docker build -t iomt-node -f "$REPO/l3_testbed/docker/Dockerfile.node" "$REPO/l3_testbed/docker"
sudo docker build -t iomt-ryu -f "$REPO/l3_testbed/docker/Dockerfile.ryu" "$REPO/l3_testbed"

step "7. TLS certificates"
cd "$REPO" && bash l3_testbed/gen_certs.sh results/l3

echo
echo "Setup finished. Next:"
echo "  sudo docker run -d --rm --name iomt-ryu --network host iomt-ryu"
echo "  sudo -E $VENV/bin/python l3_testbed/smoke_test.py"
