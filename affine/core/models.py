from __future__ import annotations
import json
import time
import textwrap
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

__version__ = "0.0.0"


def _truncate(text: Optional[str], max_len: int = 80) -> str:
    """Truncate text to max_len with ellipsis."""
    return "" if not text else textwrap.shorten(text, width=max_len, placeholder="…")


class Miner(BaseModel):
    """Miner information."""
    
    uid: int
    hotkey: str
    model: Optional[str] = None
    revision: Optional[str] = None
    block: Optional[int] = None
    chute: Optional[Dict[str, Any]] = None
    slug: Optional[str] = None
    weights_shas: Optional[list[str]] = None
    
    class Config:
        validate_assignment = True
    
    @property
    def model_dump(self):
        """Alias for dict() for pydantic v2 compatibility."""
        return self.dict


class SampleSubmission(BaseModel):
    """Sample submission from executor to API server.
    
    Minimal data structure for executor-API communication:
    - task_uuid: Identifies which task this result belongs to
    - score: Evaluation score (can be negative, environment-specific scale)
    - latency_ms: Execution time in milliseconds
    - extra: Evaluation details and metadata
    - signature: Cryptographic signature by executor wallet
    
    API server merges this with task queue data (hotkey, revision, env, task_id)
    before saving to sample_results table.
    
    Note: Score can be any float value including negative. Different environments
    may use different scales (0-1 or 0-100 or negative values). Normalization happens during scoring.
    """
    
    task_uuid: str
    score: float  # Allow any float value including negative
    latency_ms: int = Field(ge=0)
    extra: Dict[str, Any] = Field(default_factory=dict)
    signature: str = ""
    
    def get_sign_data(self) -> str:
        """Get canonical data string for signing/verification.
        
        Format: task_uuid:score:latency_ms:extra_json
        
        This ensures:
        - Deterministic signature (sorted JSON keys)
        - All critical result fields are signed
        - Metadata (hotkey, revision) comes from task queue, not signature
        """
        extra_json = json.dumps(self.extra, sort_keys=True)
        return f"{self.task_uuid}:{self.score:.6f}:{self.latency_ms}:{extra_json}"
    
    def sign(self, wallet):
        """Sign the submission with executor wallet.
        
        Args:
            wallet: Bittensor wallet with hotkey for signing
        """
        sign_data = self.get_sign_data()
        self.signature = wallet.hotkey.sign(data=sign_data.encode()).hex()
    
    def verify(self, hotkey: str) -> bool:
        """Verify submission signature.
        
        Args:
            hotkey: SS58 address of executor who signed
            
        Returns:
            True if signature is valid
        """
        try:
            import bittensor as bt
            keypair = bt.Keypair(ss58_address=hotkey)
            sign_data = self.get_sign_data()
            signature_bytes = bytes.fromhex(self.signature)
            return keypair.verify(data=sign_data.encode(), signature=signature_bytes)
        except Exception:
            return False


class Result(BaseModel):
    """Evaluation result for a miner on a specific environment.
    
    Used internally for SDK evaluation and backward compatibility.
    For executor-API communication, use SampleSubmission instead.
    
    Note: No miner object - only hotkey and revision are needed.
    Validator hotkey is stored by API server, not included in Result.
    """
    
    version: str = __version__
    
    # Miner identification (no Miner object)
    miner_hotkey: str = ""
    model_revision: str = ""
    
    # Evaluation details
    env: str
    score: float
    latency_seconds: float
    success: bool
    error: Optional[str] = None
    task_id: Optional[int] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    
    def dict(self, *args, **kwargs):
        return super().dict(*args, **kwargs)

    def json(self, **kwargs):
        return json.dumps(self.dict(**kwargs))
    
    def __repr__(self):
        return (
            f"<Result "
            f"hotkey={_truncate(self.miner_hotkey, 12)}... "
            f"env={self.env} "
            f"score={self.score:.4f}>"
        )
    
    __str__ = __repr__