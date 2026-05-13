# Affine Validator Guide

This document provides a complete guide for running a validator on the Affine subnet (Subnet 120).

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Running Methods](#running-methods)
- [CLI Reference](#cli-reference)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)

## Overview

The main responsibilities of an Affine validator are:

1. **Fetch Weights**: Get the latest normalized weights from the backend API
2. **Apply Burn Mechanism**: Allocate a percentage of weights to UID 0 based on configuration
3. **Set On-Chain**: Submit weights to the Bittensor blockchain

Validators no longer need to run evaluation or scoring logic directly. All complex computations are handled by backend services.

## Prerequisites

1. **Bittensor Wallet**: Validator hotkey registered to Subnet 120
2. **API Access**: Ability to access the Affine backend API
3. **Sufficient TAO**: To pay for on-chain transaction fees

## Environment Setup

### 1. Install Affine

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install Affine
git clone https://github.com/AffineFoundation/affine.git
cd affine
uv venv && source .venv/bin/activate && uv pip install -e .

# Verify installation
af
```

### 2. Configure Environment Variables

Copy and edit the `.env` file:

```bash
cp .env.example .env
```

Edit `.env` file with required variables:

```bash
# Bittensor wallet configuration (required)
BT_WALLET_COLD=your_coldkey_name
BT_WALLET_HOT=your_hotkey_name

# Subtensor configuration
SUBTENSOR_ENDPOINT=finney
SUBTENSOR_FALLBACK=wss://lite.sub.latent.to:443

# Validator configuration (optional)
NETUID=120                          # Subnet ID
```

### 3. Environment Variable Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `BT_WALLET_COLD` | Coldkey name | - | Yes |
| `BT_WALLET_HOT` | Hotkey name | - | Yes |
| `SUBTENSOR_ENDPOINT` | Subtensor node address | `finney` | No |
| `SUBTENSOR_FALLBACK` | Fallback Subtensor address | - | No |
| `NETUID` | Subnet ID | `120` | No |

## Running Methods

### Method 1: Docker (Recommended)

Run the validator with Docker and Watchtower for automatic updates:

```bash
# Start validator (with auto-update)
docker-compose down && docker-compose pull && docker-compose up -d && docker-compose logs -f
```

**Docker Commands:**

```bash
# Restart containers (handle OOM and other issues)
docker compose up -d --force-recreate

# View logs
docker compose logs -f

# Stop services
docker compose down

# Run with local build
docker compose -f docker-compose.yml -f docker-compose.local.yml down --remove-orphans
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build --remove-orphans
docker compose -f docker-compose.yml -f docker-compose.local.yml logs -f
```


### Method 2: Local Execution

#### Single Run Mode (Default)

Execute one weight setting and exit:

```bash
af servers validator
```

#### Service Mode (Continuous)

Set environment variable `SERVICE_MODE=true` and run:

```bash
# Set environment variable
export SERVICE_MODE=true

# Start validator service
af servers validator
```

Or specify directly on startup:

```bash
SERVICE_MODE=true af servers validator
```

#### Custom Parameters

```bash
# Specify network and wallet
af servers validator --network finney --wallet-name mywallet --hotkey-name myhotkey --netuid 120

# Use verbose logging
af -vv servers validator

# Use TRACE level logging
af -vvv servers validator
```

### Method 3: Using Systemd Service (Linux)

Create a systemd service file for auto-start and auto-restart:

```bash
# Create service file
sudo nano /etc/systemd/system/affine-validator.service
```

Service file content:

```ini
[Unit]
Description=Affine Validator Service
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/affine
Environment="PATH=/path/to/affine/.venv/bin:/usr/bin"
Environment="SERVICE_MODE=true"
EnvironmentFile=/path/to/affine/.env
ExecStart=/path/to/affine/.venv/bin/af servers validator
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable affine-validator

# Start service
sudo systemctl start affine-validator

# View status
sudo systemctl status affine-validator

# View logs
journalctl -u affine-validator -f
```

## CLI Reference

### Validator Service Command

```bash
af servers validator [OPTIONS]
```

**Options:**

- `--netuid <NETUID>`: Subnet ID (default: from `NETUID` env var or 120)
- `--wallet-name <NAME>`: Wallet name (default: from `BT_WALLET_COLD` env var)
- `--hotkey-name <NAME>`: Hotkey name (default: from `BT_WALLET_HOT` env var)
- `--network <NETWORK>`: Network name (default: from `SUBTENSOR_NETWORK` env var or finney)
- `-v, -vv, -vvv`: Increase logging verbosity (INFO, DEBUG, TRACE)

**Examples:**

```bash
# Run with environment variables
af servers validator

# Specify all parameters
af servers validator --netuid 120 --wallet-name mywallet --hotkey-name myhotkey --network finney

# Use DEBUG logging
af -vv servers validator

# Single run (default)
af servers validator

# Service mode
SERVICE_MODE=true af servers validator
```

### Query Commands

Validators can also use query commands to monitor network status:

```bash
# View latest weights
af get-weights

# View top 10 miners
af get-scores --top 10

# One-stop: window state + queue head + weights table
af get-rank

# Inspect one miner from the validator side (admin command)
AFFINE_SHOW_ADMIN_COMMANDS=true af db get-miner --uid <UID>
```

## Monitoring & Maintenance

### Runtime Status Monitoring

The validator periodically prints status information:

```
============================================================
Validator Service Status
============================================================
Running: True
Total Runs: 48
Successful: 45
Failed: 3
Success Rate: 93.8%

Weight Setter:
  Total Sets: 45
  Failed Sets: 0
  Last Set: 1732766400
============================================================
```

### Key Metrics

- **Success Rate**: Weight setting success rate, should stay above 90%
- **Total Runs**: Total number of runs
- **Failed Sets**: Number of failed weight settings

### Logging Levels

Adjust logging verbosity as needed:

```bash
# CRITICAL: Only critical errors
af servers validator

# INFO: Basic information (-v)
af -v servers validator

# DEBUG: Detailed debug information (-vv)
af -vv servers validator

# TRACE: Most detailed trace information (-vvv)
af -vvv servers validator
```

### Docker Logs

```bash
# Real-time log viewing
docker compose logs -f

# View last 100 lines
docker compose logs --tail=100

# View specific service logs
docker compose logs -f validator
```

## Troubleshooting

### Q: Validator reports "No weights available from API"

**Cause**: Backend API may be temporarily unavailable or under maintenance.

**Solutions**:
1. Check network connection
2. Verify API endpoint is accessible
3. Wait a few minutes and retry
4. Check Discord for maintenance notifications

### Q: Weight setting fails with "Failed to set weights"

**Cause**: May be on-chain transaction failure or wallet configuration error.

**Solutions**:
1. Confirm wallet configuration is correct
2. Check wallet has sufficient TAO for transaction fees
3. Verify hotkey is registered to the subnet
4. Check subtensor connection is normal
5. Use `-vv` or `-vvv` to view detailed error messages

### Q: Docker container frequently OOM (out of memory)

**Solutions**:

```bash
# Recreate containers
docker compose up -d --force-recreate

# Or use CLI
af deploy validator --recreate
```

### Q: How to verify the validator is working properly?

**Verification Steps**:

1. View logs to confirm no errors:
   ```bash
   docker compose logs -f
   ```

2. Check weights are successfully set on-chain:
   ```bash
   af get-weights
   ```

3. Monitor success rate should stay above 90%

4. Confirm your validator is active on [Taostats](https://taostats.io) or [Dashboard](https://www.affine.io/)

### Q: What hardware is required for a validator?

**Minimum Configuration**:
- **CPU**: 2 cores
- **Memory**: 4GB
- **Storage**: 20GB
- **Network**: Stable internet connection

Validators don't require GPUs as all computation is done on the backend.

### Q: What's the difference between service mode and single run mode?

**Single Run Mode (Default)**:
- Execute one weight setting and exit
- Suitable for use with cron or systemd for scheduled runs
- `SERVICE_MODE=false` or not set

**Service Mode**:
- Continuous operation, periodically setting weights at intervals
- Suitable for Docker deployment or long-term running
- `SERVICE_MODE=true`

### Q: How to update the validator?

**Docker Method**:
```bash
docker-compose down && docker-compose pull && docker-compose up -d
```

**Local Method**:
```bash
cd affine
git pull
uv pip install -e .
```

**Watchtower Auto-Update**: The Docker Compose configuration includes Watchtower, which automatically pulls the latest image and restarts services.

## Related Documentation

- [Main Documentation](../README.md) - Affine project overview
- [Miner Guide](MINER.md) - Mining guide
- [FAQ](FAQ.md) - Frequently Asked Questions
- [Bittensor Documentation](https://docs.bittensor.com/) - Official Bittensor documentation