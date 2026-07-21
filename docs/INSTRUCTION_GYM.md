# InstructionGym integration

InstructionGym is registered in affine-cortex as an evaluation-only Affinetes
environment. It is intentionally disabled by default: both
`enabled_for_sampling` and `enabled_for_scoring` are `false` in
`affine/database/system_config.json`. Local integration tests must not change
the online configuration store.

## Runtime contract

- The scheduler passes the sampled global `task_id` directly to the evaluator.
  The model `seed` controls generation only and never selects a question.
- Sampling reuses the same `random` mode as Terminal over the exact half-open
  range `[0, 541000000)`. Cortex draws unique task IDs from the flat address
  space; it does not balance templates or deduplicate materialized prompts.
  The generated task pool is persisted through the existing scheduler state
  path; public window or block values are not used as an RNG seed.
- The environment is Docker-only. Basilica is rejected before cache lookup or
  `affinetes.load_env` in SDK, miner-eval, and teacher paths.
- `UVICORN_WORKERS=1` matches the clean-tag image, SDK, validator and isolated
  E2E evidence. Raising the worker count requires a separate local concurrency,
  memory and cleanup gate before changing production registration.
- Shared validator API keys are not forwarded into the environment container.
  Teacher/logprob rollouts are rejected for InstructionGym.
- Consequently, the production model endpoint must be privately reachable
  without a shared validator application token. The local E2E may use an
  ephemeral per-call key for its loopback stub; that does not prove an
  authenticated production endpoint is usable. If production requires a token,
  add and review a least-privilege credential channel before activation.

## Local-only validation

Use an already-built local image and a model stub bound to localhost. The gate
rejects non-loopback model URLs, non-loopback Docker daemons, and pre-existing
container names in code. Do not use production credentials or the online
configuration store.

The integration gate in `scripts/test_instruction_gym_e2e.py` exercises the
real scheduler sampler, affine SDK wrapper, Affinetes container, successful
OpenAI-compatible response path, classified connection-failure path, prompt
identity, secret redaction, and cleanup. Unit coverage is in:

- `tests/test_instruction_gym_environment.py`
- `tests/test_instruction_gym_e2e_script.py`
- `tests/test_miner_eval.py`
- `tests/test_wheel_contents.py`

The test script is a local-image gate, not proof of a registry deployment.

## Production activation gates

Do not enable either runtime flag until all of the following are true:

1. The coordinated Affinetes lifecycle/validation changes are merged and
   affine-cortex's dependency and lock file pin that exact commit.
2. InstructionGym is released from a clean source tag and the evaluator image
   is published under an immutable registry digest.
3. `EnvConfig.docker_image` is changed from the mutable placeholder to that
   digest.
4. A clean host pulls the digest and passes Affinetes `afs validate` for at
   least two valid IDs and the exclusive upper boundary `541000000`.
5. The same digest passes the affine-cortex end-to-end gate without production
   credentials or online state changes.
6. The approved model endpoint is confirmed by its owner to be privately
   reachable without a shared validator application token, or a separately
   reviewed least-privilege credential channel is in place. Do not probe the
   online endpoint from this local release gate.
7. Sampling can then be enabled for a bake-in window while scoring remains
   disabled. Scoring is a separate operator decision after the accumulated
   results and infrastructure-failure rate are reviewed.

Until these gates are recorded, this repository contains integration code and
local evidence only; it does not claim an online InstructionGym rollout.
