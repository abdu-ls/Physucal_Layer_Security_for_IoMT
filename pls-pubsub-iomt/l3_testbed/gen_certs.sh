#!/usr/bin/env bash
# Self-signed CA + broker certificate for the TLS runs (baseline B3 and crypto benchmark).
# Usage: bash l3_testbed/gen_certs.sh [results/l3]   -> writes results/l3/certs/
set -euo pipefail
DIR="${1:-results/l3}/certs"
mkdir -p "$DIR"
cd "$DIR"
openssl req -x509 -newkey rsa:2048 -nodes -days 365 -keyout ca.key -out ca.crt -subj "/CN=iomt-test-ca"
openssl req -newkey rsa:2048 -nodes -keyout server.key -out server.csr -subj "/CN=iomt-broker"
printf "subjectAltName=IP:10.0.0.2,IP:10.0.0.3\n" > san.ext
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365 -extfile san.ext
chmod 644 ./*.crt ./*.key
echo "Certificates written to $DIR"
