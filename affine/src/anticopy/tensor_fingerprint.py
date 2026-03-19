"""
Tensor Fingerprinting for Anti-Copy Detection

Downloads specific tensor shards from miners' HuggingFace repos and computes
pairwise cosine similarity to detect model copies at the weight level.

This is a direct signal that cannot be evaded via inference-time perturbation —
miners would need to actually modify weights on HuggingFace, which changes
the revision hash.

Strategy: Sample 6 key tensors across the model (embed, early/mid/late layers,
lm_head), take evenly-spaced elements, compute cosine similarity.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from affine.core.setup import logger

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors import safe_open

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

# Key tensors for fingerprinting — chosen for maximum discriminative power
# across different fine-tuning strategies (full-model, top-layer, LoRA, etc.)
TARGET_TENSORS = [
    "model.embed_tokens.weight",  # embedding layer — changes with vocab tuning
    "model.layers.0.self_attn.q_proj.weight",  # first layer — early features
    "model.layers.31.self_attn.q_proj.weight",  # middle layer
    "model.layers.63.self_attn.q_proj.weight",  # last layer — most fine-tuned
    "model.layers.63.mlp.gate_proj.weight",  # last MLP — captures style changes
    "lm_head.weight",  # output projection — most sensitive to fine-tuning
]

# Elements sampled per tensor (50K * 6 tensors = 300K total per model)
SAMPLE_SIZE = 50000


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class TensorFingerprinter:
    """Downloads and caches tensor fingerprints for miners.

    Args:
        cache_dir: Directory to cache fingerprint NPZ files.
                   If None, uses a temp directory (fingerprints recomputed each run).
        hf_token:  HuggingFace API token for private repos.
        sample_size: Number of elements to sample per tensor.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        sample_size: int = SAMPLE_SIZE,
    ):
        if not HAS_DEPS:
            raise ImportError(
                "Tensor fingerprinting requires 'huggingface_hub' and 'safetensors'. "
                "Install with: pip install huggingface_hub safetensors"
            )
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.sample_size = sample_size
        self._weight_map_cache: Dict[str, Dict[str, str]] = {}

    def _find_safetensor_files(self, repo_id: str) -> List[str]:
        """List safetensor files in a HuggingFace repo."""
        try:
            files = list_repo_files(repo_id, token=self.hf_token)
            return [
                f
                for f in files
                if f.endswith(".safetensors") or f == "model.safetensors.index.json"
            ]
        except Exception as e:
            logger.debug(f"tensor_fp: error listing {repo_id}: {e}")
            return []

    def _find_tensor_shard(
        self, repo_id: str, tensor_name: str, safetensor_files: List[str]
    ) -> Optional[str]:
        """Find which shard file contains a given tensor."""
        if repo_id in self._weight_map_cache:
            return self._weight_map_cache[repo_id].get(tensor_name)

        index_file = next(
            (f for f in safetensor_files if f.endswith(".index.json")), None
        )

        if index_file:
            idx_path = hf_hub_download(repo_id, index_file, token=self.hf_token)
            with open(idx_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            self._weight_map_cache[repo_id] = weight_map
            return weight_map.get(tensor_name)
        else:
            # Single-file model
            for f in safetensor_files:
                if f.endswith(".safetensors"):
                    return f
        return None

    def _sample_tensor(
        self, repo_id: str, tensor_name: str, safetensor_files: List[str]
    ) -> Optional[np.ndarray]:
        """Download a shard and sample elements from a specific tensor."""
        shard_file = self._find_tensor_shard(repo_id, tensor_name, safetensor_files)
        if not shard_file:
            return None
        try:
            local_path = hf_hub_download(repo_id, shard_file, token=self.hf_token)
            with safe_open(local_path, framework="numpy") as f:
                if tensor_name not in f.keys():
                    return None
                tensor = f.get_tensor(tensor_name)
                flat = tensor.flatten().astype(np.float32)
                if len(flat) > self.sample_size:
                    indices = np.linspace(
                        0, len(flat) - 1, self.sample_size, dtype=int
                    )
                    return flat[indices]
                return flat
        except Exception as e:
            logger.debug(f"tensor_fp: error sampling {tensor_name} from {repo_id}: {e}")
            return None

    def fingerprint(self, repo_id: str, cache_key: str = "") -> Optional[Dict[str, np.ndarray]]:
        """Generate tensor fingerprint for a model.

        Args:
            repo_id:   HuggingFace repo ID (e.g. "user/model-name")
            cache_key: Key for caching (e.g. "uid_42" or "uid_42_rev_abc123").
                       If empty, uses repo_id hash.

        Returns:
            Dict mapping tensor name -> sampled float32 array, or None on failure.
        """
        if not cache_key:
            cache_key = repo_id.replace("/", "_")
        cache_file = self.cache_dir / f"{cache_key}.npz"

        if cache_file.exists():
            try:
                data = np.load(cache_file, allow_pickle=False)
                fp = dict(data)
                if len(fp) >= 3:
                    logger.debug(f"tensor_fp: cache hit for {cache_key} ({len(fp)} tensors)")
                    return fp
            except Exception:
                pass

        logger.info(f"tensor_fp: downloading fingerprint for {repo_id}")
        safetensor_files = self._find_safetensor_files(repo_id)
        if not safetensor_files:
            logger.warning(f"tensor_fp: no safetensor files in {repo_id}")
            return None

        fingerprint = {}
        for tensor_name in TARGET_TENSORS:
            sample = self._sample_tensor(repo_id, tensor_name, safetensor_files)
            if sample is not None:
                fingerprint[tensor_name] = sample

        if len(fingerprint) < 3:
            logger.warning(
                f"tensor_fp: only {len(fingerprint)} tensors for {repo_id}, skipping"
            )
            return None

        try:
            np.savez_compressed(cache_file, **fingerprint)
            logger.debug(f"tensor_fp: cached {len(fingerprint)} tensors for {cache_key}")
        except Exception as e:
            logger.debug(f"tensor_fp: cache write failed: {e}")

        return fingerprint

    def fingerprint_miners(
        self, miners: List[dict]
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Generate fingerprints for a list of miners.

        Args:
            miners: List of dicts with keys: uid, model, revision (optional)

        Returns:
            Dict mapping uid -> fingerprint dict
        """
        results = {}
        for miner in miners:
            uid = miner["uid"]
            repo_id = miner["model"]
            revision = miner.get("revision", "")
            cache_key = f"uid_{uid}_rev_{revision[:8]}" if revision else f"uid_{uid}"
            try:
                fp = self.fingerprint(repo_id, cache_key=cache_key)
                if fp:
                    results[uid] = fp
            except Exception as e:
                logger.warning(f"tensor_fp: failed uid={uid}: {e}")
        logger.info(
            f"tensor_fp: fingerprinted {len(results)}/{len(miners)} miners"
        )
        return results

    @staticmethod
    def compare(
        fp_a: Dict[str, np.ndarray], fp_b: Dict[str, np.ndarray]
    ) -> Tuple[float, Dict[str, float]]:
        """Compare two fingerprints.

        Returns:
            (avg_cosine, per_tensor_cosines) — avg_cosine is mean across
            all common tensors. Returns (NaN, {}) if no common tensors.
        """
        common = set(fp_a.keys()) & set(fp_b.keys())
        if not common:
            return float("nan"), {}
        per_tensor = {}
        for t in common:
            if len(fp_a[t]) == len(fp_b[t]):
                per_tensor[t] = _cosine_similarity(fp_a[t], fp_b[t])
        if not per_tensor:
            return float("nan"), {}
        avg = float(np.mean(list(per_tensor.values())))
        return avg, per_tensor

    def compare_all_pairs(
        self, fingerprints: Dict[int, Dict[str, np.ndarray]]
    ) -> Dict[Tuple[int, int], float]:
        """Compute pairwise cosine similarity for all miner pairs.

        Returns:
            Dict mapping (uid_a, uid_b) -> avg cosine similarity
            Keys are ordered so uid_a < uid_b.
        """
        uids = sorted(fingerprints.keys())
        results = {}
        for i, uid_a in enumerate(uids):
            for uid_b in uids[i + 1 :]:
                avg, _ = self.compare(fingerprints[uid_a], fingerprints[uid_b])
                if not np.isnan(avg):
                    results[(uid_a, uid_b)] = avg
        return results
