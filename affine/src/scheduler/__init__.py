"""
Queue-window scheduler.

Drives one challenger-vs-champion contest per ~7200-block window:
  - pick the earliest pending challenger
  - deploy champion + challenger on Targon
  - write per-env task assignments to system_config
  - wait for executors to fill sample_results
  - hand off to scorer (DECIDE + WEIGHT_SET)
  - tear down Targon workloads on finalize
"""
