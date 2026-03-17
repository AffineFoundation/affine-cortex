# RL Training Guide for Miners

Affine does NOT provide a built-in RL training pipeline. This is by design -- miners independently choose their training methods.

## What Affine Provides

- **Evaluation environments** (13 active) accessible via SDK
- **Scoring signals** usable as RL reward
- **Base models** downloadable from existing miners via `af pull`

## Training Approach

1. **Get base model**: Qwen3-32B (exact architecture match required)
2. **Use SDK for evaluation**: `af eval` or the Python SDK to get scores
3. **Apply RL**: Use scores as reward signal in your RL pipeline
4. **Iterate**: Improve across ALL environments (geometric mean scoring)

## Key Constraints

- Must use Qwen3-32B architecture (7-field check: hidden_size=5120, num_hidden_layers=64, etc.)
- No quantized models
- Geometric mean scoring means weakness in ANY environment collapses total score
- Pareto frontier: you need to be genuinely better, not just different

## SDK for Training

```python
import affine as af

# Evaluate against any environment
env = af.CDE()
task = env.get_task()
result = your_model(task)
score = env.score(task, result)  # Use as RL reward
```

## OpenEnv Interface (for agentic training)

```python
obs = env.reset(task_id)
while not done:
    action = your_agent(obs)
    obs, reward, done, info = env.step(action)
env.stop()
```

## Environments to Train On

Train across all 13 environments for best results. Key categories:
- **Reasoning** (ded, abd): Fast, high sample throughput
- **Code** (cde, print, swe-*): Various code generation/editing tasks
- **Logic** (lgc): Multi-step logical reasoning
- **Game** (game): Game-theoretic problems via OpenSpiel
- **Web/Nav** (liveweb, navworld): Browser and tool-use tasks
- **Other** (arc-gen, logprobs): Pattern recognition and calibration
