# Frequently Asked Questions (FAQ)

## Getting Started & General Questions

**Q1: What is the purpose of the Affine subnet?**

Affine is a Bittensor subnet designed to incentivize the creation of advanced reasoning models. The goal is to push the state-of-the-art in Reinforcement Learning and drive the development of more intelligent models by rewarding miners for improving performance on a variety of challenging tasks.

**Q2: How does mining work on this subnet?**

The validator hosts inference, so miners submit model weights rather than running serving infrastructure:

1. **Train a model.** Improve a Qwen3-based model on the subnet's evaluation environments.
2. **Upload to HuggingFace.** Push the weights to a public HuggingFace repo whose name ends with your hotkey.
3. **Commit on chain.** Run `af commit --repo user/repo-hotkey --revision <SHA>` once. Each hotkey can commit exactly one revision — any second commit permanently invalidates the miner.

The validator pulls your weights from HuggingFace, runs them through its configured inference provider, and evaluates them against the current champion when your slot comes up in the daily queue.

**Q3: What are the requirements to start mining?**

* A HuggingFace account to host your model weights.
* A registered Bittensor coldkey + hotkey.
* The validator handles inference compute for scheduled evaluations.

**Q4: Where can I find the code and leaderboard?**

* **GitHub:** [https://github.com/AffineFoundation/affine-cortex](https://github.com/AffineFoundation/affine-cortex)
* **Live Dashboard:** [https://www.affine.io/](https://www.affine.io/)

---

## How challenges work

**Q5: How does the queue work?**

Every ~7200 blocks (~24h) the scorer opens a new "window". It picks the earliest-submitted not-yet-challenged miner (ordered by `first_block` on chain) and runs both that challenger and the current champion through every environment in parallel on Targon. After all sampling completes, the comparator applies a strict **all-envs-better** rule:

- For the challenger to dethrone the champion, **every** environment's mean score must beat the champion's by at least the per-env `margin`, **and** the challenger must have collected at least `min_tasks_per_env` successful samples.
- Otherwise the challenger is permanently terminated (no second shot on the same hotkey).

**Q6: What if I lose?**

You're out for good with that hotkey. The "multi-commit rule" prevents re-trying with a new revision on the same hotkey, so once a miner is `terminated`, that hotkey can't compete again. Register a fresh hotkey to try a new model.

**Q7: What kind of tasks does the subnet evaluate?**

The environments are configured in `system_config.environments` and currently include SWE-INFINITE, LIVEWEB, NAVWORLD, MEMORY, DISTILL, TERMINAL (LOGPROBS is disabled by default). Each environment evaluates several hundred tasks per window (e.g. SWE evaluates 300 of its latest task IDs; LIVEWEB samples 400 deterministically-random task IDs).

**Q8: How does the subnet handle model copying?**

The design limits copying through three current controls:

- Public status commands show ranking, queue, weights, and miner metadata. Current-window task details stay internal to the validator.
- The challenge is sequential and bounded: each hotkey gets exactly one shot at the throne, ever. There's no leaderboard climbing by repeatedly submitting variants.
- Plagiarism detection compares `model_hash` (sha256 of weight shards): the earliest committer wins; any later miner with an identical hash is marked `invalid` by the monitor.

---

## Troubleshooting

**Q9: My model committed but doesn't show in `af get-rank`. Why?**

Two likely reasons:

- The monitor refresh hasn't run since you committed (refreshes every ~5 min). `af get-rank` includes the queue head — wait one refresh and re-check.
- The monitor's `_validate_miner` rejected your commit. Common reasons:
  - HuggingFace can't resolve your repo at that revision (404)
  - Repo name doesn't end with your hotkey (after block 7,290,000)
  - Multi-commit rule: you committed twice (any hotkey with `commit_count > 1` past block 7,710,000 is permanently invalid)
  - Plagiarism: another miner submitted the same `model_hash` earlier

**Q10: My slot finally came up but my evaluation failed. What happened?**

Run `af get-rank` to see whether you appear as the current battle's challenger. Two common failure modes:

- **Provider allocation failed.** The validator couldn't get your model loaded within the model-load timeout. Your model may be too large, the HF download timed out, or the inference server rejected your config. You're marked `terminated` with a deployment failure reason and the queue advances.
- **Inference unreachable.** The workload launched but per-task requests time out or 5xx. The executor still writes a sample row per task (with score 0 + error). Once all rows are in, the comparator sees the challenger far below the champion in every env and you're marked `terminated` with a loss reason.

---

## Subnet mechanics

**Q11: How is the weight allocated?**

Winner-takes-all. The current champion holds `overall_score=1.0`; every other miner is at `0.0`. The validator writes that to chain once per window via `af servers validator`.

**Q12: Why are emissions sometimes burned to UID 0?**

The validator can be configured with a `validator_burn_percentage` (in `system_config`) which routes that fraction of weight to UID 0. Used to ensure fairness and prevent exploiters from profiting while the team rolls out fixes, mitigates downtime, or handles network instability.

**Q13: How can I see the live state?**

```
af get-rank      # one-stop rank/status table
af get-scores    # top-N scores from the latest snapshot
af get-score 42  # one miner's score
af get-weights   # latest normalized weights (what the validator sets on chain)
```

`af get-rank` is the only status command you usually need; everything else
is a narrow helper.

There is no per-miner "stats" command and no way to query sample data through the API — that's intentional. Yesterday's samples may be published by an external frontend, but this codebase never serves them.
