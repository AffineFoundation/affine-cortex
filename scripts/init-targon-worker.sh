#!/usr/bin/env bash
#
# One-shot bootstrap for a fresh Targon K8s-pod GPU worker.
#
# Usage (per host):
#   ssh wrk-XXX@ssh.deployments.targon.com 'bash -s' < init-targon-worker.sh
#
# Override the pre-pulled sglang image with SGLANG_IMAGE=...
#
# Idempotent: safe to re-run.

set -uo pipefail

SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:latest}"

ok()   { printf '\e[32m✓\e[0m %s\n' "$*"; }
warn() { printf '\e[33m!\e[0m %s\n' "$*"; }
die()  { printf '\e[31m✗\e[0m %s\n' "$*" >&2; exit 1; }
hd()   { printf '\n=== %s ===\n' "$*"; }

# --- 0. sanity --------------------------------------------------------------

hd "0. sanity"

[ "$EUID" -eq 0 ] || die "must run as root"
command -v docker >/dev/null   || die "docker not installed"
command -v nvidia-smi >/dev/null || die "nvidia-smi not present (no driver?)"
command -v nvidia-ctk >/dev/null || die "nvidia-ctk missing (need NCT >= 1.14)"

gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
ok "root, docker $(docker --version | awk '{print $3}' | tr -d ,), NCT $(nvidia-container-toolkit --version | head -1 | awk '{print $NF}'), $gpu_count GPUs"

# --- 1. /data ---------------------------------------------------------------

hd "1. /data"

if [ ! -d /data ]; then
  mkdir -p /data
  ok "created /data"
else
  ok "/data already exists"
fi

free_gb=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 100 ] || warn "/data has only ${free_gb}G free — model downloads need ~70G each"

if ! touch /data/.bootstrap_probe 2>/dev/null; then
  die "/data not writable"
fi
rm -f /data/.bootstrap_probe
ok "/data writable, ${free_gb}G free"

# --- 2. no-cgroups (the K8s-pod fix) ----------------------------------------

hd "2. nvidia-container-cli: no-cgroups = true"

CFG=/etc/nvidia-container-runtime/config.toml
[ -f "$CFG" ] || die "$CFG missing"

if grep -q "^no-cgroups = true" "$CFG"; then
  ok "already set"
elif grep -q "^#no-cgroups = false" "$CFG"; then
  sed -i 's|^#no-cgroups = false|no-cgroups = true|' "$CFG"; ok "uncommented + flipped to true"
elif grep -q "^no-cgroups = false" "$CFG"; then
  sed -i 's|^no-cgroups = false|no-cgroups = true|' "$CFG"; ok "flipped false → true"
else
  sed -i '/^\[nvidia-container-cli\]/a no-cgroups = true' "$CFG"
  ok "inserted no-cgroups = true under [nvidia-container-cli]"
fi

# --- 3. CDI spec (belt + braces — works if anyone uses --device nvidia.com/gpu=all) ---

hd "3. /etc/cdi/nvidia.yaml"

if [ -s /etc/cdi/nvidia.yaml ]; then
  ok "CDI spec already present ($(wc -c </etc/cdi/nvidia.yaml) bytes)"
else
  mkdir -p /etc/cdi
  nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml >/dev/null 2>&1 \
    && ok "generated CDI spec" \
    || warn "nvidia-ctk cdi generate failed (CDI mode won't work, --gpus all still will)"
fi

# --- 4. pre-pull sglang image ----------------------------------------------

hd "4. pre-pull $SGLANG_IMAGE"

if docker image inspect "$SGLANG_IMAGE" >/dev/null 2>&1; then
  ok "image already cached locally"
else
  docker pull "$SGLANG_IMAGE" 2>&1 | tail -3
  docker image inspect "$SGLANG_IMAGE" >/dev/null 2>&1 \
    && ok "pulled" \
    || die "docker pull failed"
fi

# --- 5. smoke test: --gpus all actually maps GPUs into a container ----------

hd "5. smoke test: --gpus all"

probe_out=$(docker run --rm --gpus all "$SGLANG_IMAGE" nvidia-smi -L 2>&1 || true)
seen=$(printf '%s\n' "$probe_out" | grep -c '^GPU [0-9]' || true)

if [ "$seen" -eq "$gpu_count" ]; then
  ok "container sees all $gpu_count GPUs via --gpus all"
elif [ "$seen" -gt 0 ]; then
  warn "container sees $seen / $gpu_count GPUs"
else
  printf '%s\n' "$probe_out" | head -10
  die "no GPUs visible inside container — bootstrap failed"
fi

hd "done"
ok "host ready for affine sglang deployment"
