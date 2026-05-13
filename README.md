# Affine

Mine open reasoning.

[Affine Discord](https://discord.com/invite/3T9X4Yn23e) | [Live Dashboard](https://www.affine.io/)

## Introduction

Affine is an incentivized RL environment that pays miners who make incremental improvements on a set of tasks (such as program abduction or coding). The mechanism is sybil-proof, decoy-proof, copy-proof, and overfitting-proof.

**How does Affine work?**

Affine validators incentivize miners to submit models on Bittensor. Miners commit a HuggingFace `(model, revision)` pair on chain; validators host the inference (currently via Targon, optionally an operator-managed B300 fleet). The validator-side scheduler walks the queue of pending miners in `first_block` order — each one faces the current champion across every evaluation environment in a single back-to-back contest. The challenger only dethrones the champion when they win **strictly** across all environments by a per-env margin; otherwise they're permanently terminated and the queue advances to the next miner. Every ~7200 blocks (~24h) the per-env task-id pool is refreshed and the (current) champion is re-sampled before the queue continues. The winner-takes-all weight goes to the champion until they're dethroned.

**Why Affine?** 

Directed incentives for RL have never been achieved. The ability to direct intelligence and aggregate the work effort of a large, non-permissioned group of individuals on RL tasks will unlock rapid advancement in intelligence. We intend to commoditize reasoning (intelligence's highest form) and break the intelligence sound barrier.

## Installation

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

### Architecture

Affine uses [Affinetes](https://github.com/AffineFoundation/affinetes) for container orchestration, providing:
- Clean, lightweight container management
- Support for local and remote Docker deployments
- Environment caching for improved performance
- Type-safe environment definitions

All evaluation environments are packaged as pre-built Docker images, eliminating the need for complex sandbox management.

## Getting Started

### For Miners

📖 **[Complete Miner Guide →](docs/MINER.md)**

Learn how to:
- Set up your environment and configure API keys
- Pull models from the network
- Improve models with reinforcement learning
- Upload to HuggingFace and commit on-chain (validator hosts inference)
- Use CLI commands to query public rank/status and miner metadata

### For Validators

📖 **[Complete Validator Guide →](docs/VALIDATOR.md)**

Learn how to:
- Set up and configure your validator
- Run with Docker (recommended) or locally
- Monitor validator performance
- Troubleshoot common issues
- Set weights on-chain

### Additional Resources

- 📚 **[FAQ](docs/FAQ.md)** - Frequently asked questions

## SDK Usage

Affine can be used as an SDK for evaluating models across different environments.

**Examples:**
- [`examples/sdk.py`](examples/sdk.py) - Evaluate miners from the network on DED-V2 and ABD-V2 environments
- [`examples/sdk2.py`](examples/sdk2.py) - Evaluate custom models by specifying model parameters directly

**Key Features:**
- Evaluate registered miners by UID
- Evaluate custom models with direct parameters (model, base_url, temperature)
- Support for multiple environments (DED-V2, ABD-V2, etc.)
- List all available environments
- Async API for efficient batch evaluation

See the example files for complete usage patterns.

## Support

- **Discord**: [Join our community](https://discord.com/invite/3T9X4Yn23e)
- **Dashboard**: [https://www.affine.io/](https://www.affine.io/)
- **GitHub**: [https://github.com/AffineFoundation/affine-cortex](https://github.com/AffineFoundation/affine-cortex)
- **FAQ**: [docs/FAQ.md](docs/FAQ.md)
