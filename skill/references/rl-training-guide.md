# RL Training Guide for Miners

> Verified against affine-cortex@e3a9fce (2026-06).

Affine does NOT provide a built-in RL training pipeline. This is by design — miners independently choose their training methods. The subnet only provides evaluation environments and a reward signal.

## What Affine Provides

- **Evaluation environments** accessible via SDK / `af eval`
- **Scoring signals** usable as RL reward
- **Base models** downloadable from existing miners via `af pull`

The subnet now **self-hosts inference** for evaluation — you don't deploy anywhere. You train, upload to HuggingFace, and commit `{model, revision}` on-chain.

## Target: beat the champion across the 4 scoring envs

Scoring is **Champion Challenge** (winner-takes-all), not geometric-mean/ELO. The only thing that matters: dethrone the current champion. To win you must be **dominant in ≥ 1 scoring env AND not-worse (≥ 98%) in every scoring env** — a regression in any one env is an automatic loss. So train to *not regress anywhere*, then push hard on at least one env.

The 4 active scoring environments:

| Env | What it tests | Notes |
|-----|---------------|-------|
| **SWE-INFINITE** | SWE-bench-style coding (real repos, DOOD) | sampling_count 200; heavily anti-hack hardened |
| **NAVWORLD** | Travel planning / tool use (AMap), LLM-judged | sampling_count 300 |
| **MEMORY** | Memory management (intake/store/retrieve) | sampling_count 100; long episodes |
| **TERMINAL** | CLI/terminal agent in a sandbox (DOOD) | sampling_count 300 |

DISTILL-V2 is sampled but **not scored** (shadow). Older envs (ded/abd/cde/print/lgc/game/swe-pro/arc/liveweb/logprobs) are no longer in the live contest.

## Model constraints (STRICT)

- Must match an allowed architecture **exactly**:
  - **Qwen3-32B** (dense): `model_type qwen3`, hidden_size 5120, num_hidden_layers 64, etc.
  - **Qwen3.6-35B-A3B** (MoE, multimodal): `model_type qwen3_5_moe`, language fields under `text_config.*`, num_experts 256.
  - A block gate (`AFFINE_QWEN36_ONLY_ENFORCE_BLOCK`) can restrict *new* submissions to Qwen3.6 only.
- **No quantized models** (any `quantization_config` → rejected).
- **No gated/private repos** (validator must download your weights to serve them).
- HF repo **name must contain "affine"** and **end with your hotkey**.
- **One commit per hotkey, ever** — validate exhaustively before committing; a new attempt needs a fresh hotkey.

## SDK for evaluation

```python
import affine as af

env = af.SWE_INFINITE()   # or af.MEMORY(), af.NAVWORLD(), af.TERMINAL()
task = env.get_task()
result = your_model(task)
score = env.score(task, result)   # use as RL reward
```

## OpenEnv interface (agentic training)

```python
obs = env.reset(task_id)
while not done:
    action = your_agent(obs)
    obs, reward, done, info = env.step(action)
env.stop()
```

Used mainly by SWE-INFINITE / TERMINAL (multi-step tool use).

## Practical loop

1. **Get a baseline**: `af pull <champion-uid> --model-path ./model` to start from the current best.
2. **Train**: apply your RL/SFT method; reward = env scores.
3. **Benchmark vs champion**: `af eval --env <ENV> --uid <champion-uid> --samples 20` per env; make sure you don't regress anywhere.
4. **Commit once**: `af miner-deploy --repo myuser/affine-<...>-<hotkey> -p ./model`.
