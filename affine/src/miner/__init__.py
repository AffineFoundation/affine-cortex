"""
Miner-side commands.

Provides the CLI surface miners use to participate:
  - pull          : fetch a UID's committed model from HuggingFace
  - commit        : write {model, revision} on chain (one-shot per hotkey)
  - miner-deploy  : convenience wrapper (HF upload → commit)
  - get-rank      : one-stop status (window + queue + weights)
  - get-weights / get-scores / get-score
"""

__version__ = "1.0.0"
