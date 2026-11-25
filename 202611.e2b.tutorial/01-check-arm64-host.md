# Step 1 — Check the ARM64 Host

This step verifies that an Ubuntu 24.04 ARM64 server has the basic host capabilities required before deploying the complete E2B self-hosted infrastructure.

This step is intentionally **non-destructive**, except that the automated script may install Ubuntu's `cpu-checker` package if `kvm-ok` is unavailable.

## 1. Check CPU architecture

```bash
uname -m
```

Expected:

```text
aarch64
```

`aarch64` confirms that the host is running a 64-bit ARM architecture.

## 2. Check Ubuntu version

```bash
cat /etc/os-release
```

Confirm:

```text
ID=ubuntu
VERSION_ID="24.04"
```

## 3. Check CPU

```bash
lscpu
```

Useful summary:

```bash
lscpu | grep -E 'Architecture|CPU\(s\)|Model name|Vendor ID|Virtualization'
```

Also check the number of logical CPUs:

```bash
nproc
```

## 4. Check memory

```bash
free -h
```

The `available` column is more useful than `free` for estimating currently usable memory.

## 5. Check KVM

E2B sandboxes use Firecracker microVMs, so KVM availability is a critical host requirement.

```bash
ls -l /dev/kvm
```

Expected: `/dev/kvm` exists.

Also check loaded KVM modules:

```bash
lsmod | grep -i kvm
```

If `/dev/kvm` does not exist, stop here and investigate host virtualization before continuing.

## 6. Check KVM permissions

```bash
stat /dev/kvm
getent group kvm
id
```

If the current user does not have access through the `kvm` group:

```bash
sudo usermod -aG kvm "$USER"
```

Then log out/in again, or use:

```bash
newgrp kvm
```

## 7. Run the KVM diagnostic

Install the Ubuntu diagnostic utility:

```bash
sudo apt update
sudo apt install -y cpu-checker
```

Then:

```bash
sudo kvm-ok
```

On ARM64, do not treat `kvm-ok` as the only authority because the utility is primarily designed around x86 virtualization checks. `/dev/kvm` and an actual ARM64 Firecracker test are more meaningful.

## 8. Check kernel

```bash
uname -a
uname -r
```

Record the kernel version because it can matter when diagnosing KVM, Firecracker, and networking issues.

## 9. Check disk

```bash
df -h /
df -ih /
```

E2B will eventually need storage for sandbox images/root filesystems, Firecracker artifacts, logs, Nomad data, and build artifacts.

## 10. Check network interfaces

```bash
ip -br addr
```

Record the primary interface and host IP. Later E2B networking will depend on the host's actual interface configuration.

## 11. Check routing

```bash
ip route
```

Verify that the server has an appropriate default route.

Do not change routes in Step 1.

## 12. Check DNS

```bash
resolvectl status
getent hosts github.com
```

DNS must work before downloading E2B source code and dependencies.

## 13. Check HTTPS connectivity

```bash
curl -I https://github.com
```

A successful HTTP response confirms basic outbound HTTPS connectivity.

## 14. Check basic tools

```bash
command -v git
command -v curl
command -v wget
command -v unzip
command -v jq
command -v make
command -v gcc
command -v g++
command -v python3
```

These tools will be used by later installation steps.

---

# Automated check

The accompanying script performs the checks above:

```bash
chmod +x 01-check-arm64-host.sh
./01-check-arm64-host.sh
```

The script does not install E2B, Nomad, Consul, Firecracker, Terraform, or Docker, and it does not modify routing, IP forwarding, bridges, NAT, or systemd services.

## Pass criteria

Before proceeding to Step 2, the important checks are:

- `uname -m` → `aarch64`
- Ubuntu → `24.04`
- `/dev/kvm` exists
- KVM is available
- adequate CPU and memory
- adequate disk space
- working network
- working DNS
- working outbound HTTPS
- Git and basic build utilities available

If `/dev/kvm` is missing, **stop before proceeding to the E2B deployment steps**.
