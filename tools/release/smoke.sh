#!/usr/bin/env bash
# Install a built wheel into a fresh virtualenv (hash-locked deps) and exercise it from
# outside the source tree. Release tooling; not shipped.
#
#   tools/release/smoke.sh WHEEL LOCK            # no Semgrep: expect explicit PARTIAL/degraded
#   tools/release/smoke.sh WHEEL LOCK semgrep    # user-style Semgrep install: expect full/complete
#
# NO_NETWORK=1 (Linux only) runs the scan inside a network namespace with no interfaces
# except loopback, proving the deterministic scan needs no network.
set -euo pipefail

WHEEL=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
LOCK=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
MODE=${3:-none}
PY=${PYTHON:-python3}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

"$PY" -m venv venv
V="$WORK/venv/bin"
"$V/python" -m pip install --quiet --no-input --upgrade pip
"$V/python" -m pip install --quiet --no-input --require-hashes -r "$LOCK"
"$V/python" -m pip install --quiet --no-input --no-deps "$WHEEL"

if [ "$MODE" = "semgrep" ]; then
  # External dependency, installed the way a user would (not part of the lock or wheel).
  "$V/python" -m pip install --quiet --no-input "semgrep>=1.90,<2"
  SEMGREP="$V/semgrep"
  export PATH="$V:$PATH"
  EXPECT_DOCTOR=0 EXPECT_SCAN=0 EXPECT_STATUS=complete
else
  SEMGREP="$WORK/no-semgrep-here"
  EXPECT_DOCTOR=4 EXPECT_SCAN=4 EXPECT_STATUS=degraded
fi
export SEMGREP_SEND_METRICS=off SEMGREP_ENABLE_VERSION_CHECK=0

"$V/tantular-cyber" --version
"$V/tantular-cyber" --help >/dev/null
"$V/tantular-cyber" plugins list >/dev/null
"$V/tantular-cyber" sample --dest "$WORK/demo"

set +e
"$V/tantular-cyber" doctor --json --no-ollama --semgrep-bin "$SEMGREP" >doctor.json
DOCTOR=$?
SCAN_CMD=("$V/tantular-cyber" scan --repo "$WORK/demo" --output "$WORK/reports"
  --semgrep-bin "$SEMGREP")
if [ "${NO_NETWORK:-0}" = "1" ] && [ "$(uname -s)" = "Linux" ]; then
  # Root is needed only to create the namespace; the scan itself runs as the invoking user
  # (same HOME, no root-owned files left behind for later Semgrep runs).
  sudo -E unshare --net -- bash -c 'ip link set lo up 2>/dev/null; user=$1; shift;
    exec sudo -E -u "$user" -- "$@"' _ "$(id -un)" "${SCAN_CMD[@]}"
  SCAN=$?
else
  "${SCAN_CMD[@]}"
  SCAN=$?
fi
set -e

"$V/python" - "$DOCTOR" "$EXPECT_DOCTOR" "$SCAN" "$EXPECT_SCAN" "$EXPECT_STATUS" <<'EOF'
import json, sys
doctor, want_doctor, scan, want_scan, want_status = sys.argv[1:]
d = json.load(open("doctor.json"))
r = json.load(open("reports/report.json"))
assert doctor == want_doctor, f"doctor exit {doctor} != {want_doctor}"
assert d["network_requests"] == [], d["network_requests"]
assert scan == want_scan, f"scan exit {scan} != {want_scan}"
assert r["status"] == want_status, r["status"]
assert r["schema_version"] == "1.3.0", r["schema_version"]
assert r["llm"]["enabled"] is False
assert r["totals"]["findings"] > 0, "sample project must produce findings"
print(f"doctor={d['scan_capability']} scan={r['status']} findings={r['totals']['findings']}")
EOF

if [ "$MODE" = "semgrep" ]; then
  if ! "$V/tantular-cyber" benchmark --output "$WORK/bench"; then
    "$V/python" - "$WORK/bench/scan-report.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print("benchmark scan status:", r["status"], r["status_reasons"])
for e in r["errors"]:
    print("  error:", e["stage"], e["message"][:400])
EOF
    exit 1
  fi
fi
echo "smoke OK ($MODE)"
