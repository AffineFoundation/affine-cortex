# Miner Guide

How to participate in Affine as a miner.

> The validator now hosts inference for you. You commit a HuggingFace
> `(model, revision)` pair on chain and the validator-side scheduler
> pulls your weights into Targon when your queue slot comes up. There
> is no Chutes deployment, no GPU rental, no inference server to keep
> warm.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Mining Workflow](#mining-workflow)
- [CLI Reference](#cli-reference)
- [Multi-commit Rule](#multi-commit-rule)
- [Tips](#tips)

## Prerequisites

1. **Bittensor wallet** — coldkey + hotkey, registered on subnet 120.
2. **HuggingFace account** with a personal access token that has *Write*
   scope (you'll upload model weights through it).

That's all. No Chutes account, no GPU funding.

## Environment Setup

### 1. Install Affine

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/AffineFoundation/affine-cortex.git
cd affine-cortex
uv venv && source .venv/bin/activate && uv pip install -e .
af --help
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in:

```bash
# Bittensor wallet aliases (names, not SS58 addresses)
BT_WALLET_COLD=mywallet
BT_WALLET_HOT=myhotkey

# Subtensor endpoint
SUBTENSOR_ENDPOINT="finney"

# HuggingFace token (Write scope; starts with hf_...)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Register on subnet 120

```bash
btcli subnet register --netuid 120
```

## Mining Workflow

### Step 1 — Get a baseline model

Pull a recent on-chain model as your starting point:

```bash
af pull <UID> --model-path ./model_path
```

This reads UID's commit, fetches the model from HuggingFace at the committed
revision, and saves to the local path. Iterate on this checkpoint.

### Step 2 — Improve

Train, fine-tune, distill — whatever produces a model that beats the
current champion on every evaluation environment. Validate locally with
`af eval --env <ENV> --base-url <local-vllm> --model <name>` (see [CLI
Reference](#cli-reference)).

### Step 3 — Upload to HuggingFace and commit

One command does both:

```bash
af miner-deploy --repo myuser/affine-model-<hotkey> -p ./model_path
```

This:
1. Uploads `./model_path` to `myuser/affine-model-<hotkey>` on HuggingFace
   (creates the repo if needed).
2. Reads the resulting commit SHA.
3. Writes `{"model": "myuser/affine-model-<hotkey>", "revision": "<SHA>"}`
   on chain via `bittensor.set_reveal_commitment`.

The repo name **must end with your hotkey** (case-insensitive) after
block 7,290,000 — the monitor rejects miners that don't follow this rule.

If you already uploaded out of band:

```bash
af commit --repo myuser/affine-model-<hotkey> --revision <SHA>
```

`af commit` is a single-shot: each hotkey can commit **exactly once**. Any
second commit invalidates the miner permanently (see [Multi-commit Rule](#multi-commit-rule)).

## CLI Reference

### Status queries

| Command | What it shows |
| --- | --- |
| `af get-rank` | One-stop public rank/status table |
| `af get-miner --uid UID` | Public miner metadata: model, revision, validity, queue status, and commit blocks |
| `af get-weights` | Latest on-chain-bound weights only |
| `af get-scores --top N` | Top N miners from the latest snapshot |
| `af get-score <UID>` | One miner's score |

`af get-rank` is the main status command; the others are focused query
helpers. `af get-miner` exposes public miner metadata.

### Local evaluation

`af eval` is a developer tool to sanity-check a model against an Affine
environment without going through the validator. You bring up an
OpenAI-compatible inference server (vllm/sglang locally, or any
OpenAI-compatible host) and point at it:

```bash
af eval --env affine:ded-v2 \
        --base-url http://localhost:8000/v1 \
        --model myuser/affine-model \
        --samples 10
```

`af eval --list-envs` enumerates the available environments.

### Service commands

Miners don't run any of the backend services; those are for validators.

## Multi-commit Rule

**The single biggest gotcha.** Once a hotkey commits a revision, that
hotkey cannot commit again. The monitor's validator checks
`commit_count > 1` and marks the miner `permanently invalid` with
reason `multiple_commits:count=N`. You will be excluded from the
challenger queue and never recover with that hotkey.

If you need to try a different model, register a fresh hotkey.

## Tips

- **Repo name ends with hotkey** — case-insensitive. Easiest way: name
  your HF repo `affine-model-<hotkey-suffix>`.
- **Repo must be public** when you commit — the validator pulls weights
  via the public HuggingFace API.
- **Model size** is checked by the monitor (`check_model_size`). Stick
  with Qwen3-32B-class models; oversized models are rejected at
  validation time.
- **Chat template safety** is also checked (`check_template_safety`).
  Don't ship chat templates with arbitrary code execution; they will
  trip the `malicious_template:*` invalidation.
- **Patience.** Each window is ~24h and processes one challenger.
  `af get-rank` shows the queue head; once you're at the top of the
  queue, expect your turn within the next window.
