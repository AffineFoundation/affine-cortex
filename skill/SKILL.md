# Affine — Agent Skill Document

> Affine is an incentivized RL subnet on Bittensor (SN120). Miners train reasoning models, validators evaluate them, rewards are distributed via on-chain weights. Tagline: *"Mine open reasoning."*

**Key insight**: Miners don't run mining hardware. They train models (Qwen3-32B), upload to HuggingFace, deploy inference on Chutes.ai, and commit metadata on-chain.

**Repos**:
- [affine-cortex](https://github.com/AffineFoundation/affine-cortex) — Main codebase (services, SDK, CLI, scoring)
- [affinetes](https://github.com/AffineFoundation/affinetes) — Container orchestration
- [liveweb-arena](https://github.com/AffineFoundation/liveweb-arena) — Browser-based web task environment
- [affine-dash-starter](https://github.com/AffineFoundation/affine-dash-starter) — React dashboard

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [How to Mine](#2-how-to-mine)
3. [How to Validate](#3-how-to-validate)
4. [Environments](#4-environments)
5. [Scoring Pipeline](#5-scoring-pipeline)
6. [Anti-Cheat Mechanisms](#6-anti-cheat-mechanisms)
7. [SDK Usage](#7-sdk-usage)
8. [CLI Command Reference](#8-cli-command-reference)
9. [Development Guide](#9-development-guide)
10. [Testing](#10-testing)
11. [Key Code Paths](#11-key-code-paths)
12. [Environment Variables](#12-environment-variables)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. System Architecture

### 1.1 Overview

6 backend microservices running as Docker containers on an internal `affine-backend` network, backed by DynamoDB (PAY_PER_REQUEST):

```
┌─────────────────────────────────────────────────────────────────┐
│                        Affine System                            │
│                                                                 │
│  ┌─────────┐   ┌───────────┐   ┌──────────┐   ┌────────────┐  │
│  │  Miner   │   │ Validator  │   │ Executor │   │  Scorer    │  │
│  │ (train+  │   │ (set      │   │ (run     │   │ (4-stage   │  │
│  │  deploy) │   │  weights)  │   │  tasks)  │   │  algo)     │  │
│  └────┬─────┘   └─────┬─────┘   └────┬─────┘   └─────┬──────┘  │
│       │               │              │               │          │
│  ┌────▼───────────────▼──────────────▼───────────────▼──────┐   │
│  │                    API Server (FastAPI)                    │   │
│  │   /tasks/fetch, /tasks/submit, /samples/scoring, etc.     │   │
│  └────────────────────────┬──────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────▼──────────────────────────────────┐   │
│  │              DynamoDB (PAY_PER_REQUEST)                    │   │
│  │  Tables: sample_results, task_pool, scores, miners, ...   │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────┐  ┌────────────┐  ┌────────────────────────┐   │
│  │  Scheduler   │  │  Monitor   │  │  AntiCopy Detector     │   │
│  │ (task gen +  │  │ (miner     │  │  (hidden states +      │   │
│  │  rotation)   │  │  tracking) │  │   logprob cosine)      │   │
│  └──────────────┘  └────────────┘  └────────────────────────┘   │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  Affinetes (Container Orchestration)                      │   │
│  │  Docker / SSH / Basilica modes                            │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Services

| Service | Container | Memory | Purpose | Command |
|---------|-----------|--------|---------|---------|
| **API** | `affine-api` | 2-4GB | FastAPI hub — task fetch/submit, scoring endpoints | `af servers api` |
| **Scheduler** | `affine-scheduler` | 2-4GB | Task generation + FIFO rotation | `af servers scheduler` |
| **Executor** | `affine-executor` | 4-8GB | Runs evaluations in Docker containers (DOOD) | `af servers executor` |
| **Monitor** | `affine-monitor` | 2-4GB | 12-step miner validation pipeline, every 5min | `af servers monitor` |
| **Scorer** | `affine-scorer` | 4-8GB | 4-stage weight calculation | `af servers scorer` |
| **AntiCopy** | `affine-anticopy` | 2-4GB | Behavioral plagiarism detection | `af servers anticopy` |

Plus **Watchtower** for auto-updating all containers every 30s.

### 1.3 API Routes (prefix: `/api/v1`)

| Router | Key Endpoints | Purpose |
|--------|---------------|---------|
| **tasks** | `POST /tasks/fetch`, `POST /tasks/submit` | Task queue management |
| **samples** | `GET /samples/{hotkey}/{env}/{task_id}`, `GET /samples/scoring` | Evaluation results + scoring data |
| **miners** | `GET /miners/uid/{uid}`, `GET /miners/uid/{uid}/stats` | Miner info + stats |
| **scores** | `GET /scores/latest`, `GET /scores/weights/latest` | Score snapshots + weights |
| **config** | System config endpoints | Environment configs |

### 1.4 Task Flow

```
Executor                    API (TaskPoolManager)
   │                              │
   ├──POST /tasks/fetch──────────►│  1. Get ALL pending tasks for env
   │   X-Hotkey, X-Signature     │  2. Random shuffle (anti-starvation)
   │                              │  3. Take first batch_size tasks
   │◄──TaskFetchResponse─────────│  4. Assign to executor
   │                              │
   │  [Execute eval in Docker]    │
   │                              │
   ├──POST /tasks/submit─────────►│  1. Verify executor signature
   │   SampleSubmission           │  2. Save sample, log, delete task
   │◄──SampleSubmitResponse──────│  3. Return immediately
```

### 1.5 Container Orchestration (Affinetes)

Lightweight framework for running eval environments in Docker containers.

| Mode | Description |
|------|-------------|
| **docker** (default) | Local or remote Docker via SSH |
| **url** | Connect to user-deployed HTTP service |
| **basilica** | K8s pods with TTL auto-cleanup |

Each environment is a Docker image with a FastAPI server. Affinetes auto-injects HTTP:
- `GET /health` — health check
- `POST /call` — execute `{"method": "evaluate", "args": [], "kwargs": {...}}`

```bash
# Affinetes CLI
afs init my-env --template actor      # Initialize environment
afs build my-env --tag my-env:v1      # Build Docker image
afs run my-env:v1 --name my-env       # Start container
afs call my-env evaluate --arg task_id=10  # Call method
afs validate my-env --num-tests 100   # Validate seed consistency
```

---

## 2. How to Mine

### 2.1 Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.11+ |
| Package manager | `uv` (recommended) |
| Bittensor wallet | Coldkey + hotkey with TAO for registration |
| Chutes account | Same hotkey as Bittensor, funded with TAO |
| HuggingFace account | Write-access token |
| GPU (optional) | For local model training/fine-tuning |

### 2.2 Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/AffineFoundation/affine-cortex.git
cd affine-cortex
uv venv && source .venv/bin/activate && uv pip install -e .

# Verify
af --help
```

### 2.3 Wallet & Registration

```bash
# Create wallet
btcli wallet new_coldkey --wallet.name mywallet
btcli wallet new_hotkey --wallet.name mywallet --wallet.hotkey myhotkey

# Register to subnet
btcli subnet register --wallet.name mywallet --wallet.hotkey myhotkey

# Register Chutes (MUST use same hotkey)
chutes register
cat ~/.chutes/config.ini   # shows payment address — fund it
```

### 2.4 Environment Configuration

```bash
cp .env.example .env
```

Edit `.env`:
```bash
BT_WALLET_COLD=mywallet
BT_WALLET_HOT=myhotkey
SUBTENSOR_ENDPOINT=finney
SUBTENSOR_FALLBACK=wss://lite.sub.latent.to:443
CHUTES_API_KEY=cpk_xxxxx
CHUTE_USER=myusername
HF_TOKEN=hf_xxxxxxx
```

### 2.5 The Mining Pipeline

```
1. Pull existing model   →  af pull <UID> --model-path ./my_model
2. Improve with RL       →  (your own training pipeline)
3. Upload to HuggingFace →  huggingface-cli upload <repo> ./my_model
4. Deploy to Chutes      →  af chutes_push --repo <repo> --revision <SHA>
5. Commit on-chain       →  af commit --repo <repo> --revision <SHA> --chute-id <id>
```

Or one command: `af miner-deploy -r <repo> -p ./my_model`

### 2.6 One-Command Deploy

```bash
# Public repo
af miner-deploy -r myuser/my-model -p ./my_model

# Private repo (recommended — prevents copying before commit)
af miner-deploy -r myuser/my-model -p ./my_model --private-repo

# Dry run
af miner-deploy -r myuser/my-model -p ./my_model --dry-run
```

The `--private-repo` workflow:
1. Creates PRIVATE HuggingFace repo
2. Uploads model privately
3. Stores HF_TOKEN as Chutes secret
4. Commits to blockchain FIRST
5. Makes repo PUBLIC after commit confirmed
6. Deploys to Chutes

### 2.7 Manual Deploy (Step by Step)

```bash
# 1. Upload model to HuggingFace
huggingface-cli upload myuser/my-model ./my_model

# 2. Deploy to Chutes → returns JSON with chute_id
af chutes_push --repo myuser/my-model --revision <hf-commit-sha>

# 3. Commit on-chain
af commit --repo myuser/my-model --revision <hf-commit-sha> --chute-id <chute-id>
```

### 2.8 Evaluate Locally (Before Deploying)

```bash
# List environments
af eval --list-envs

# Eval against local model server
af eval --env GAME --base-url http://localhost:8000/v1 --model my-model --samples 10 --network-host

# Eval against existing miner (benchmark)
af eval --env GAME --uid 7 --samples 10

# Eval specific task
af eval --env GAME --uid 7 --task-id 502284834

# Eval task range
af eval --env GAME --uid 7 --task-id-range 100 110

# Debug output
af -vv eval --env GAME --uid 7 --samples 5
```

### 2.9 Monitor Your Miner

```bash
af get-miner <your-uid>       # Miner info
af get-rank                    # Full ranking table
af get-score <your-uid>        # Your score
af get-scores --top 20         # Top miners
af get-sample <uid> <env> <task-id>  # Specific sample result
af get-pool <uid> <env>        # Pending tasks
af get-weights                 # Current weights
af get-envs                    # Environment configs
```

### 2.10 Key Rules for Miners

1. **Model naming**: Must contain "affine", HuggingFace repo must end with your hotkey
2. **Architecture**: Must be exact Qwen3-32B (7 fields checked: hidden_size=5120, num_hidden_layers=64, etc.)
3. **No quantized models** allowed
4. **No cheating in chat_template**: LLM audit checks for built-in solvers
5. **No copying**: Hidden states + logprob cosine similarity detection (threshold 0.99)
6. **Diversify**: Geometric mean scoring — can't just optimize one environment
7. **Stay online**: Chute must be "hot" — offline miners get 0 score

### 2.11 Tips for Competitive Mining

1. Use `--private-repo` to prevent copying before commit
2. Keep Chutes warm — increase `shutdown_after_seconds` or send periodic requests
3. Evaluate before deploying — benchmark against top miners with `af eval`
4. Use negative UIDs: syntax `n1` for UID -1 (e.g., `af pull n1`)
5. Monitor validator activity with `af get-pool`
6. Iterate with RL — Pareto frontier means you need to be *better* than existing models

---

## 3. How to Validate

### 3.1 Hardware Requirements

| Resource | Minimum |
|----------|---------|
| **CPU** | 2 cores |
| **Memory** | 4 GB |
| **Storage** | 20 GB |
| **GPU** | **Not required** — all evaluation happens on the backend |

### 3.2 Installation & Setup

```bash
git clone https://github.com/AffineFoundation/affine-cortex.git
cd affine-cortex
uv sync  # or pip install -e .

# Create wallet
btcli wallet new_coldkey --wallet.name my_validator
btcli wallet new_hotkey --wallet.name my_validator --wallet.hotkey default

# Register on subnet
btcli subnet register --netuid 120 --wallet.name my_validator --wallet.hotkey default
```

### 3.3 Environment Configuration

```bash
# .env file
BT_WALLET_COLD=my_validator
BT_WALLET_HOT=default
SUBTENSOR_ENDPOINT=finney
SUBTENSOR_FALLBACK=wss://lite.sub.latent.to:443
NETUID=120
WEIGHT_SET_INTERVAL_BLOCKS=180
SERVICE_MODE=true
```

### 3.4 Running the Validator

**CLI:**
```bash
af servers validator --wallet-name my_validator --hotkey-name default

# With verbose logging
af -vv servers validator --wallet-name my_validator --hotkey-name default

# Full options
af servers validator \
  --netuid 120 \
  --wallet-name my_validator \
  --hotkey-name default \
  --network finney \
  --watchdog-timeout 600
```

**Docker (recommended for production):**
```bash
af deploy validator           # Remote mode
af deploy validator --local   # Local
af deploy validator --recreate  # Recreate container
af deploy validator --restart   # Restart existing
```

Docker deployment includes Watchtower for automatic updates.

### 3.5 Validator Internal Loop

```
1. Wait for weight submission window (every 180 blocks)
2. Fetch weights from API (up to 12 retries, 3s apart)
3. Fetch config (burn_percentage)
4. Process weights:
   - Parse uid → weight mappings
   - Handle system miners (uid > 1000) → redirect to UID 0
   - Apply burn percentage
   - Normalize to sum = 1.0
   - Enforce 1% min threshold
   - Redistribute sub-threshold to UID 0
5. Set weights on-chain (3 retries, waits for finalization)
6. Sleep 10s, repeat
```

A **watchdog** thread monitors for stuck blocks — if no new block for 600s, process self-restarts via SIGTERM.

### 3.6 Monitoring

```bash
af get-weights              # Weights your validator would set
af get-scores --top 10      # Top miners
af get-score 42             # Specific miner
af get-rank                 # Full ranking table
af get-envs                 # Environment configs

# Docker logs
docker logs -f affine-validator
docker ps | grep affine
```

### 3.7 Validator Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Watchdog restart loop | Subtensor node unresponsive | Check `SUBTENSOR_ENDPOINT`, try fallback |
| "No weights from API" | Backend API down | Check network; 12 retries happen automatically |
| Weight submission fails | Insufficient stake | Check with `btcli wallet balance`; re-register |
| Block stuck >600s | Node sync issue | Restart validator, switch endpoint |

---

## 4. Environments

13 containerized Docker environments managed by Affinetes:

| Environment | Docker Image | Timeout | Category |
|-------------|-------------|---------|----------|
| **Deduction** (`ded-v2`) | `affine:ded-v2` | 600s | Reasoning |
| **Abduction** (`abd-v2`) | `affine:abd-v2` | 600s | Reasoning |
| **CDE** (`cde`) | `affine:cde` | 600s | Code |
| **Print** (`print`) | `affine:print` | 600s | Code |
| **Logic** (`lgc` / `lgc-v2`) | `affine:lgc-v2` | 1200-1800s | Logic |
| **Game** (`game`) | `affine:game` | 7200s | Game (OpenSpiel) |
| **SWE-Pro** (`swe-pro`) | `affine:swe-pro` | 1800s | Code (SWE-bench) |
| **SWE-Synth** (`swe-synth`) | `affine:swe-synth` | 7200s | Code (SWE-bench) |
| **SWE-Infinite** (`swe-infinite`) | `affine:swe-infinite` | 7200s | Code (SWE-bench) |
| **LiveWeb** (`liveweb`) | `affine:liveweb` | 1200s | Web/Browser |
| **NavWorld** (`navworld`) | `affine:navworld` | 7200s | Navigation (MCP tools) |
| **ARC-GEN** (`arc-gen`) | `affine:arc-gen` | 600s | Pattern recognition |
| **LogProbs** (`logprobs`) | `affine:logprobs` | 600s | Calibration |

Each environment runs as a Docker container with an `Actor.evaluate()` method returning `{"score": 0.0-1.0}`.

### Environment Patterns

| Pattern | Examples | Characteristics |
|---------|----------|----------------|
| **Simple Deterministic** | print, ded, abd | Single model call, fast (<60s), many workers |
| **Multi-step Reasoning** | lgc, cde | Complex problems, 1200-1800s timeout |
| **Agentic/Interactive** | swe-pro, liveweb | Multiple tool calls, Docker-in-Docker, 1800-7200s |
| **Game/Adversarial** | game | Game-theoretic, CPU-bound, 7200s |
| **Tool-augmented** | navworld | MCP tools, external APIs, LLM-judge scoring |

---

## 5. Scoring Pipeline

4-stage pipeline in `affine/src/scorer/`:

### Stage 1: Data Collection (`stage1_collector.py`)
- Fetch scoring data from API (`/api/v1/samples/scoring`)
- Calculate average scores per environment per miner
- Validate completeness (>=90% of required samples)

### Stage 2: Pareto Filtering (`stage2_pareto.py`)
- **Anti-plagiarism**: prevents model copies from earning rewards
- Sort miners by `first_block` (earlier = priority)
- For each environment subset, compare miner pairs on **common tasks**
- Gap formula: `gap = clamp(z_score × SE, MIN_IMPROVEMENT, MAX_IMPROVEMENT)` where `SE = sqrt(p(1-p)/n)`
- Miner B must beat threshold in **ALL** environments to dominate A
- Dominated miners are filtered from subset

### Stage 3: Subset Scoring (`stage3_subset.py`)
- Geometric mean across environments (with epsilon=0.01 smoothing)
- Rank-based decay: `adjusted = score × 0.5^(rank-1)`
- Distribute subset weight proportionally

### Stage 4: Weight Normalization (`stage4_weights.py`)
- Sum subset weights per miner
- Remove miners below 1% threshold
- Normalize to sum = 1.0
- Redistribute sub-threshold to UID 0

### Key Scoring Parameters (`scorer/config.py`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `Z_SCORE` | 1.5 | Statistical confidence (~87%) |
| `MIN_IMPROVEMENT` | 0.02 | 2% minimum gap for dominance |
| `MAX_IMPROVEMENT` | 0.10 | 10% cap on dominance threshold |
| `DECAY_FACTOR` | 0.5 | Rank decay — 50% per position |
| `GEOMETRIC_MEAN_EPSILON` | 0.01 | Smoothing to prevent zero collapse |
| `MIN_WEIGHT_THRESHOLD` | 0.01 | 1% minimum weight |
| `MIN_COMPLETENESS` | 0.9 | 90% completeness required |

---

## 6. Anti-Cheat Mechanisms

| Mechanism | Protects Against | How It Works |
|-----------|-----------------|-------------|
| **Pareto dominance** | Sybil attacks | Copies must beat original everywhere; earlier `first_block` wins |
| **AntiCopy** | Model plagiarism | Two-signal voting: hidden state cosine + logprob top-3 cosine (threshold 0.99) |
| **Weight hashing** | Direct model cloning | SHA256 of safetensors |
| **Geometric mean** | Gaming one environment | Score collapses if ANY env is near zero |
| **Task rotation** | Overfitting/memorization | FIFO pools, random sampling, billion-range task IDs |
| **Commit-reveal** | Front-running | Two-phase on-chain commitment |

### AntiCopy Detail

File: `affine/src/anticopy/detector.py`

- Two-signal voting: hidden-state cosine similarity + logprob top-3 similarity
- Threshold: 0.99 on **both** signals
- Detection requires ALL signals to agree (conservative)
- Models detected as copies have weights zeroed
- Runs as a 24h loop service

---

## 7. SDK Usage

```python
import affine as af

# Create environment instance
env = af.CDE()  # or af.GAME(), af.SWE_PRO(), af.DEDUCTION(), etc.

# Get a task
task = env.get_task()

# Evaluate your model
result = your_model(task)

# Score it (use as RL reward signal)
score = env.score(task, result)
```

Available SDK environments match the 13 active environments. The SDK provides the training interface for miners to evaluate their models locally.

### OpenEnv Training Interface

For RL training, environments expose a gym-like interface:
```python
# reset() → initial observation
# step(action) → (observation, reward, done, info)
# stop() → cleanup
```

Used primarily by SWE-INFINITE and complex agentic environments.

---

## 8. CLI Command Reference

### Miner Commands

| Command | Purpose |
|---|---|
| `af miner-deploy -r <repo> -p <path>` | Full pipeline: upload → Chutes → commit |
| `af pull <uid>` | Download model from existing miner |
| `af chutes_push --repo <repo> --revision <SHA>` | Deploy model to Chutes |
| `af commit --repo <repo> --revision <SHA> --chute-id <id>` | Submit commitment on-chain |
| `af eval` | Evaluate models against environments |

### Query Commands

| Command | Purpose |
|---|---|
| `af get-miner <uid>` | Query miner information |
| `af get-rank` | Full miner ranking table |
| `af get-scores [--top N]` | Top miners by score |
| `af get-score <uid>` | Specific miner's score |
| `af get-weights` | Normalized weights |
| `af get-sample <uid> <env> <task_id>` | Evaluation sample result |
| `af get-pool <uid> <env>` | Miner's task pool |
| `af get-envs` | Environment configurations |

### Admin/Server Commands

Hidden unless `AFFINE_SHOW_ADMIN_COMMANDS=true`:

| Command | Purpose |
|---|---|
| `af servers api/executor/monitor/anticopy/scorer/scheduler/validator` | Start backend services |
| `af deploy validator/backend/api [--local] [--recreate] [--restart]` | Deploy Docker services |
| `af down validator/backend/api [--local] [--volumes]` | Stop services |
| `af db` | Database management |
| `af miner-stats get-stats <hotkey> <revision>` | Historical stats |
| `af miner-stats list-all [--limit N]` | List all historical miners |

### Global Options

- `-v` / `-vv` / `-vvv` — Increase logging verbosity (INFO/DEBUG/TRACE)
- All commands support wallet options: `--wallet-name`, `--hotkey-name`

---

## 9. Development Guide

### 9.1 Adding a New Environment

**An Affine environment is:**
1. A Docker image exposing `/evaluate` endpoint (FastAPI on port 8000)
2. A registry entry in `affine/core/environments.py`
3. A scoring config in `affine/database/system_config.json`

**Step 1: Build Docker Image**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`POST /evaluate` request:
```json
{
  "model": "org/model-name",
  "base_url": "https://slug.chutes.ai/v1",
  "task_id": 42,
  "seed": 123456789,
  "temperature": 0.0,
  "timeout": 600
}
```

Response:
```json
{
  "score": 0.85,
  "success": true,
  "error": null,
  "extra": {"steps": 12, "detail": "metadata"}
}
```

Requirements:
- **Deterministic**: Same `(task_id, seed, model)` → reproducible results
- **Stateless**: Each `/evaluate` call is independent
- **Health check**: `GET /health` returning 200

**Step 2: Register in Environment Registry**

Edit `affine/core/environments.py`:
```python
# In _ENV_CONFIGS_CANONICAL:
"my-env": EnvConfig(
    name="my-env",
    docker_image="affinefoundation/my-env:latest",
    env_vars={"UVICORN_WORKERS": "10"},
    eval_params={"temperature": 0.0, "timeout": 600},
),

# In _ENV_ALIASES:
"MY-ENV": "my-env",
```

**EnvConfig fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | required | Canonical name |
| `docker_image` | str | required | Docker Hub image |
| `env_type` | str | `"affine"` | Type identifier |
| `env_vars` | Dict | `{}` | Container env vars |
| `required_env_vars` | List | `[]` | Required host env vars |
| `mem_limit` | str | `"10g"` | Container memory limit |
| `volumes` | Dict | None | Docker volume mounts |
| `eval_params` | Dict | `{temperature: 0.0, timeout: 600}` | Default eval params |
| `proxy_timeout` | int | 600 | HTTP proxy timeout |
| `cpu_limit` | str | None | CPU limit (Basilica mode) |

**Step 3: Add Scoring Config**

Edit `affine/database/system_config.json`:
```json
{
  "environments": {
    "my-env": {
      "enabled_for_sampling": true,
      "enabled_for_scoring": true,
      "min_completeness": 0.9,
      "sampling_config": {
        "dataset_range": [[0, 1000000000]],
        "sampling_count": 200,
        "rotation_enabled": true,
        "rotation_count": 3,
        "rotation_interval": 3600,
        "scheduling_weight": 1.0
      }
    }
  }
}
```

Load config:
```bash
python -m affine.database.cli load-config affine/database/system_config.json
```

**Step 4: Publish & Test**
```bash
docker build -t affinefoundation/my-env:latest .
docker push affinefoundation/my-env:latest
af eval --env my-env --base-url http://localhost:8000/v1 --model my-model --samples 5 --network-host
```

### 9.2 New Environment Checklist

- [ ] Docker image built and pushed
- [ ] `/evaluate` accepts `{model, base_url, task_id, seed, ...}` → `{score, success, error, extra}`
- [ ] `/health` returns 200
- [ ] `EnvConfig` added to `environments.py`
- [ ] Aliases added to `_ENV_ALIASES`
- [ ] Scoring config in `system_config.json`
- [ ] Config loaded via CLI
- [ ] `af eval` passes locally
- [ ] Task generation is deterministic given `(task_id, seed)`
- [ ] Anti-cheat: large task space, rotation, no fingerprinting
- [ ] Resource limits tuned

### 9.3 Architecture Decisions

- **DynamoDB over SQL**: PAY_PER_REQUEST billing, no connection pool management
- **Docker-in-Docker (DOOD)**: SWE-bench envs mount `/var/run/docker.sock` for nested containers
- **Basilica mode**: K8s pods with TTL for auto-cleanup, used for heavy environments (GAME)
- **Deterministic seeds**: SHA256-based, ensures reproducible evaluations
- **Signed submissions**: Executor wallet signs `(task_uuid, score, latency_ms, extra_json)` to prevent tampering

---

## 10. Testing

### Running Tests

```bash
# From affine-cortex root
pytest tests/

# Specific test file
pytest tests/test_scorer.py -v

# With coverage
pytest --cov=affine tests/
```

### Local Environment Testing

```bash
# Evaluate against a local model server
af eval --env <ENV> --base-url http://localhost:8000/v1 --model <model-name> --samples <N> --network-host

# Validate environment seed consistency (via affinetes)
afs validate my-env --num-tests 100
```

---

## 11. Key Code Paths

### affine-cortex (main repo)

| File | Purpose |
|------|---------|
| `affine/cli/main.py` | CLI entry point — all `af` commands |
| `affine/core/environments.py` | Environment configs + SDKEnvironment class |
| `affine/core/models.py` | Miner, Result, SampleSubmission models |
| `affine/core/sampling_list.py` | Sampling list management |
| `affine/core/sdk.py` | Python SDK |
| `affine/api/server.py` | FastAPI server with lifespan management |
| `affine/api/routers/tasks.py` | Task fetch/submit endpoints |
| `affine/api/routers/samples.py` | Sample results + scoring endpoint |
| `affine/api/routers/scores.py` | Score/weights endpoints |
| `affine/api/routers/miners.py` | Miner info endpoints |
| `affine/api/services/task_pool.py` | TaskPoolManager — random selection + UUID cache |
| `affine/api/services/scoring_cache.py` | Proactive scoring data cache |
| `affine/api/services/auth.py` | Bittensor signature verification |
| `affine/src/scorer/scorer.py` | 4-stage scoring orchestrator |
| `affine/src/scorer/stage1_collector.py` | Stage 1: Data collection |
| `affine/src/scorer/stage2_pareto.py` | Stage 2: Pareto filtering |
| `affine/src/scorer/stage3_subset.py` | Stage 3: Subset scoring |
| `affine/src/scorer/stage4_weights.py` | Stage 4: Normalization |
| `affine/src/scorer/config.py` | Scoring parameters |
| `affine/src/executor/main.py` | ExecutorManager (multiprocess) |
| `affine/src/executor/worker.py` | Task execution worker |
| `affine/src/anticopy/detector.py` | Copy detection (two-signal voting) |
| `affine/src/anticopy/main.py` | AntiCopy service (24h loop) |
| `affine/src/monitor/miners_monitor.py` | Miner validation (12 steps) |
| `affine/src/validator/main.py` | Validator weight-setting logic |
| `affine/src/validator/weight_setter.py` | Weight processing + on-chain submission |
| `affine/src/scheduler/main.py` | Scheduler service |
| `affine/src/scheduler/sampling_scheduler.py` | Per-miner sampling + scheduling |
| `affine/src/scheduler/task_generator.py` | Task generation |
| `affine/database/schema.py` | DynamoDB table definitions (9 tables) |
| `affine/database/dao/*.py` | Data access objects |
| `compose/docker-compose.backend.yml` | Production backend (6 services) |
| `docker-compose.yml` | Validator deployment |

### affinetes (container orchestration)

| File | Purpose |
|------|---------|
| `affinetes/api.py` | Public API: `load_env()`, `build_image_from_env()` |
| `affinetes/core/wrapper.py` | EnvironmentWrapper — dynamic method dispatch |

### liveweb-arena

| File | Purpose |
|------|---------|
| Browser-based web task evaluation | LiveWeb environment implementation |

---

## 12. Environment Variables

| Variable | Required By | Purpose |
|---|---|---|
| `CHUTES_API_KEY` | All envs | Chutes.ai API authentication |
| `HF_TOKEN` | Monitor, SWE-bench | HuggingFace model access |
| `HF_USER` | Monitor | HuggingFace username for naming checks |
| `AMAP_MAPS_API_KEY` | NavWorld | AMap API for travel planning |
| `COINGECKO_API_KEY` | LiveWeb | CoinGecko API for crypto tasks |
| `DOCKER_HUB_USERNAME` | SWE-bench Synth/Infinite | Docker Hub for nested images |
| `DOCKER_HUB_TOKEN` | SWE-bench Synth/Infinite | Docker Hub auth |
| `AFFINETES_MODE` | SDK | `docker` or `basilica` |
| `AFFINETES_HOSTS` | SDK | Comma-separated host list |
| `AFFINE_META_CONCURRENCY` | SDK | Miner query concurrency (default: 12) |
| `SUBTENSOR_ENDPOINT` | Validator/SDK | Bittensor RPC endpoint |
| `SUBTENSOR_FALLBACK` | Validator | Fallback: `wss://lite.sub.latent.to:443` |
| `AFFINE_SHOW_ADMIN_COMMANDS` | CLI | Show admin commands (`true`) |
| `BT_WALLET_COLD` | All | Bittensor coldkey wallet name |
| `BT_WALLET_HOT` | All | Bittensor hotkey name |
| `NETUID` | Validator | Subnet ID (default: 120) |
| `WEIGHT_SET_INTERVAL_BLOCKS` | Validator | Blocks between weight submissions (default: 180) |
| `SERVICE_MODE` | Validator | Continuous operation (`true`) |

---

## 13. Troubleshooting

### Miner Issues

| Issue | Fix |
|-------|-----|
| Model rejected | Ensure exact Qwen3-32B architecture (7 fields), name contains "affine", repo ends with hotkey |
| Chute offline | Keep chute warm — increase `shutdown_after_seconds`, send periodic requests |
| Zero score | Check completeness across ALL environments; geometric mean collapses on any zero |
| Detected as copy | Ensure model is genuinely different; check AntiCopy thresholds (0.99 both signals) |
| Commit failed | Verify wallet has TAO, is registered to subnet, check `SUBTENSOR_ENDPOINT` |

### Validator Issues

| Issue | Fix |
|-------|-----|
| Watchdog restart loop | Check `SUBTENSOR_ENDPOINT`, try fallback endpoint |
| No weights from API | Backend API down; 12 retries happen automatically |
| Weight submission fails | Check stake with `btcli wallet balance`; re-register if needed |
| Block stuck >600s | Restart validator, switch subtensor endpoint |

### General

| Issue | Fix |
|-------|-----|
| `af` command not found | Run `pip install -e .` from affine-cortex root |
| DynamoDB errors | Check AWS credentials and region configuration |
| Docker socket errors | Ensure `/var/run/docker.sock` is accessible (executor needs DOOD) |

---

*For detailed reference material, see [skill/references/](./references/)*
*Source: Distilled from [UNDERSTANDING.md](../UNDERSTANDING.md) (~8340 lines)*
