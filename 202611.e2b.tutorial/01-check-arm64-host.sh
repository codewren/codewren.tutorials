#!/usr/bin/env bash
set -u

PASS=0
WARN=0
FAIL=0

pass() { echo "[PASS] $1"; PASS=$((PASS + 1)); }
warn() { echo "[WARN] $1"; WARN=$((WARN + 1)); }
fail() { echo "[FAIL] $1"; FAIL=$((FAIL + 1)); }
section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

section "1. CPU Architecture"
ARCH="$(uname -m)"
echo "Architecture: $ARCH"
if [ "$ARCH" = "aarch64" ]; then
    pass "Host architecture is ARM64 (aarch64)."
else
    fail "Expected aarch64, detected: $ARCH"
fi

section "2. Operating System"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "Distribution : ${NAME:-unknown}"
    echo "Version      : ${VERSION:-unknown}"
    echo "Version ID   : ${VERSION_ID:-unknown}"
    if [ "${ID:-}" = "ubuntu" ] && [ "${VERSION_ID:-}" = "24.04" ]; then
        pass "Ubuntu 24.04 detected."
    else
        warn "This host is not detected as Ubuntu 24.04."
    fi
else
    fail "/etc/os-release does not exist."
fi

section "3. CPU Information"
lscpu | grep -E 'Architecture|CPU\(s\)|Model name|Vendor ID|Virtualization' || true
echo
echo "Logical CPUs: $(nproc)"

section "4. Memory"
free -h

section "5. KVM Device"
if [ -e /dev/kvm ]; then
    ls -l /dev/kvm
    pass "/dev/kvm exists."
else
    fail "/dev/kvm does not exist."
fi

section "6. KVM Kernel Modules"
if lsmod 2>/dev/null | grep -qi kvm; then
    lsmod | grep -i kvm
    pass "KVM kernel module appears to be loaded."
else
    warn "No loaded KVM module was detected."
fi

section "7. KVM Permissions"
if [ -e /dev/kvm ]; then
    KVM_GROUP="$(stat -c '%G' /dev/kvm 2>/dev/null || true)"
    echo "/dev/kvm group: ${KVM_GROUP:-unknown}"
    if [ -n "$KVM_GROUP" ] && id -nG "$USER" | tr ' ' '\n' | grep -qx "$KVM_GROUP"; then
        pass "Current user belongs to the /dev/kvm group."
    else
        warn "Current user may not have direct access to /dev/kvm."
        echo "If needed: sudo usermod -aG $KVM_GROUP \$USER"
    fi
fi

section "8. KVM Diagnostic"
if ! command -v kvm-ok >/dev/null 2>&1; then
    echo "kvm-ok not found; installing cpu-checker..."
    if command -v apt-get >/dev/null 2>&1; then
        if sudo apt-get update && sudo apt-get install -y cpu-checker; then
            pass "cpu-checker installed."
        else
            warn "Could not install cpu-checker."
        fi
    fi
fi
if command -v kvm-ok >/dev/null 2>&1; then
    sudo kvm-ok 2>&1 || true
    echo "NOTE: kvm-ok is primarily x86-oriented; /dev/kvm and actual ARM64 Firecracker testing are more authoritative."
fi

section "9. Kernel"
uname -a
echo
uname -r

section "10. Disk Space"
df -h /
echo
df -ih /

section "11. Network Interfaces"
ip -br addr

section "12. Routing"
ip route

section "13. DNS"
if command -v resolvectl >/dev/null 2>&1; then
    resolvectl status | sed -n '1,100p'
fi
if getent hosts github.com >/dev/null 2>&1; then
    getent hosts github.com
    pass "DNS resolution works."
else
    fail "DNS resolution for github.com failed."
fi

section "14. Internet Connectivity"
if command -v curl >/dev/null 2>&1 && curl -fsSI --connect-timeout 10 https://github.com >/dev/null 2>&1; then
    pass "HTTPS connectivity to github.com works."
else
    fail "Could not connect to https://github.com."
fi

section "15. Required Basic Tools"
TOOLS=(git curl wget unzip jq make gcc g++ python3)
for tool in "${TOOLS[@]}"; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo "[PASS] $tool: $(command -v "$tool")"
        PASS=$((PASS + 1))
    else
        echo "[WARN] $tool: not installed"
        WARN=$((WARN + 1))
    fi
done

section "Summary"
echo "PASS: $PASS"
echo "WARN: $WARN"
echo "FAIL: $FAIL"
echo

if [ "$FAIL" -gt 0 ]; then
    echo "RESULT: HOST CHECK FAILED"
    exit 1
elif [ "$WARN" -gt 0 ]; then
    echo "RESULT: HOST CHECK PASSED WITH WARNINGS"
    exit 0
else
    echo "RESULT: HOST CHECK PASSED"
    exit 0
fi
