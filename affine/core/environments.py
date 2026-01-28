#!/usr/bin/env python3

import os
import time
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from affine.core.models import Result
from affine.core.setup import logger
import affinetes as af_env


# ========================= Global Cache =========================

_ENV_CACHE: Dict[str, Any] = {}
_ENV_LOCK = Lock()


# ========================= Utility Functions =========================

def convert_memory_format(mem_limit: str, mode: str) -> str:
    """Convert memory format between Docker and Kubernetes formats.
    
    Docker format: 10g, 8g, 512m
    Kubernetes format: 10Gi, 8Gi, 512Mi
    
    Args:
        mem_limit: Memory limit string
        mode: Execution mode ('docker' or 'basilica')
        
    Returns:
        Converted memory limit string
        
    Examples:
        >>> convert_memory_format("10g", "docker")
        "10g"
        >>> convert_memory_format("10g", "basilica")
        "10Gi"
        >>> convert_memory_format("512m", "basilica")
        "512Mi"
    """
    if mode == "basilica":
        # Convert Docker format to Kubernetes format
        if mem_limit.endswith("g"):
            return mem_limit.replace("g", "Gi")
        elif mem_limit.endswith("m"):
            return mem_limit.replace("m", "Mi")
    return mem_limit


# ========================= Configuration =========================

@dataclass
class EnvConfig:
    """Environment-specific configuration"""
    name: str
    docker_image: str
    env_type: str = "affine"
    env_vars: Dict[str, str] = field(default_factory=dict)
    # Environment variables that must be sourced from the host process environment
    # and forwarded into the container (e.g., credentials/endpoints).
    required_env_vars: List[str] = field(default_factory=list)
    mem_limit: str = "10g"
    volumes: Optional[Dict[str, Dict[str, str]]] = None
    eval_params: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.0,
        "timeout": 600,
    })
    proxy_timeout: int = 600
    
    # Basilica mode configuration (optional)
    cpu_limit: Optional[str] = None  # e.g., "4000m" for basilica mode


# ========================= Environment Configurations =========================

# Canonical environment configurations
_ENV_CONFIGS_CANONICAL = {
    "affine:ded-v2": EnvConfig(
        name="affine:ded-v2",
        docker_image="affinefoundation/affine-env:v4",
        env_vars={"UVICORN_WORKERS": "10"},
        eval_params={
            "task_type": "ded",
            "temperature": 0.0,
            "timeout": 600,
        },
    ),
    "affine:abd-v2": EnvConfig(
        name="affine:abd-v2",
        docker_image="affinefoundation/affine-env:v4",
        env_vars={"UVICORN_WORKERS": "10"},
        eval_params={
            "task_type": "abd",
            "temperature": 0.0,
            "timeout": 600,
        },
    ),
    
    # PrimeIntellect environments (no task_type)
    "cde": EnvConfig(
        name="cde",
        docker_image="affinefoundation/cde:pi",
        mem_limit="25g",
        env_vars={"UVICORN_WORKERS": "4"},
        eval_params={
            "temperature": 0.0,
            "timeout": 600,
        },
    ),
    "lgc": EnvConfig(
        name="lgc",
        mem_limit="20g",
        docker_image="affinefoundation/lgc:pi",
        env_vars={"UVICORN_WORKERS": "15"},
        eval_params={
            "temperature": 0.0,
            "timeout": 1200,
        },
        proxy_timeout=1300,
    ),
    "lgc-v2": EnvConfig(
        name="lgc-v2",
        mem_limit="20g",
        docker_image="affinefoundation/lgc:pi-v2",
        env_vars={"UVICORN_WORKERS": "30"},
        eval_params={
            "temperature": 0.0,
            "timeout": 600,
        },
        proxy_timeout=620,
    ),
    "game": EnvConfig(
        name="game",
        docker_image="affinefoundation/game:openspiel",
        env_vars={"UVICORN_WORKERS": "50"},
        eval_params={
            "temperature": 0.0,
            "timeout": 7200,
        },
        proxy_timeout=7400,
        cpu_limit="2000m",
        mem_limit="8g",
    ),
    
    # SWE-bench Pro environment (requires DOOD)
    "swe-pro": EnvConfig(
        name="swe-pro",
        docker_image="affinefoundation/swebench:pro",
        env_type="swebench",
        env_vars={"UVICORN_WORKERS": "10"},
        mem_limit="10g",
        volumes={
            "/var/run/docker.sock": {
                "bind": "/var/run/docker.sock",
                "mode": "rw"
            }
        },
        eval_params={
            "max_iterations": 200,
            "temperature": 0.0,
            "timeout": 1800,
        },
        proxy_timeout=2000,
    ),
    # SWE-bench Synth environment (requires R2 credentials for dataset/artifact access)
    "swe-synth": EnvConfig(
        name="swe-synth",
        docker_image="affinefoundation/swebench:synth",
        env_type="swebench",
        env_vars={"UVICORN_WORKERS": "10"},
        required_env_vars=[
            "R2_ENDPOINT_URL",
            "R2_BUCKET",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
        ],
        mem_limit="10g",
        volumes={
            "/var/run/docker.sock": {
                "bind": "/var/run/docker.sock",
                "mode": "rw"
            }
        },
        eval_params={
            "max_iterations": 30,
            "temperature": 0.0,
            "timeout": 7200,
        },
        proxy_timeout=7300,
    ),
    "print": EnvConfig(
        name="print",
        docker_image="affinefoundation/cde:print",
        env_vars={"UVICORN_WORKERS": "15"},
        eval_params={
            "temperature": 0.0,
            "timeout": 600,
        },
    ),
    # ARC-GEN environment
    "arc-gen": EnvConfig(
        name="arc-gen",
        docker_image="affinefoundation/arc:latest",
        env_vars={"UVICORN_WORKERS": "10"},
        eval_params={
            "temperature": 0.0,
            "timeout": 600,
            "num_train": 3,
        },
        proxy_timeout=620,
    ),
}

# Alias mappings (multiple names can map to the same canonical config)
_ENV_ALIASES = {
    # ABD aliases - all point to v2
    "affine:abd": "affine:abd-v2",
    "abd": "affine:abd-v2",
    "abd-v2": "affine:abd-v2",
    
    # DED aliases - all point to v2
    "affine:ded": "affine:ded-v2",
    "ded": "affine:ded-v2",
    "ded-v2": "affine:ded-v2",
    
    # SAT aliases
    "sat": "affine:sat",
    
    # PrimeIntellect aliases (uppercase versions)
    "CDE": "cde",
    "LGC": "lgc",
    "LGC-V2": "lgc-v2",
    "LGC-v2": "lgc-v2",
    "GAME": "game",
    
    # SWE-bench aliases
    "SWE-PRO": "swe-pro",
    "SWE-SYNTH": "swe-synth",
    
    # Print aliases
    "PRINT": "print",

    # ARC-GEN aliases
    "ARC-GEN": "arc-gen",
    "ARCGEN": "arc-gen",
}

# Build final ENV_CONFIGS with aliases
ENV_CONFIGS = {}
for canonical_name, config in _ENV_CONFIGS_CANONICAL.items():
    ENV_CONFIGS[canonical_name] = config

# Add all aliases
for alias, canonical in _ENV_ALIASES.items():
    if canonical in _ENV_CONFIGS_CANONICAL:
        ENV_CONFIGS[alias] = _ENV_CONFIGS_CANONICAL[canonical]


# ========================= Base Environment =========================

class SDKEnvironment:
    """Unified SDK environment implementation"""
    
    def __init__(self, env_name: str, mode: Optional[str] = None):
        """Initialize SDK environment
        
        Args:
            env_name: Environment name
            mode: Execution mode override ('docker' or 'basilica').
                  If not specified, will use mode from affinetes_hosts.json or default to docker.
        """
        if env_name not in ENV_CONFIGS:
            raise ValueError(f"Unknown environment: {env_name}")
        
        self.config = ENV_CONFIGS[env_name]
        self._mode_override = mode
        self._env = self._load_environment()
        self._env_lock = asyncio.Lock()
    
    @property
    def env_name(self) -> str:
        return self.config.name
    
    @property
    def env_type(self) -> str:
        return self.config.env_type
    
    @property
    def docker_image(self) -> str:
        return self.config.docker_image
    
    def _get_env_vars(self) -> Dict[str, str]:
        """Get environment variables for this environment"""
        api_key = os.getenv("CHUTES_API_KEY")
        if not api_key:
            raise ValueError("CHUTES_API_KEY environment variable is required")
        
        env_vars = {"CHUTES_API_KEY": api_key}

        # Forward any required host env vars into the container for this environment
        for key in self.config.required_env_vars:
            value = os.getenv(key)
            if not value:
                raise ValueError(
                    f"{key} environment variable is required for environment '{self.env_name}'"
                )
            env_vars[key] = value
        
        # Add ENV_NAME for affine environments (from task_type in eval_params)
        if "task_type" in self.config.eval_params:
            env_vars["ENV_NAME"] = self.config.eval_params["task_type"]
        
        env_vars.update(self.config.env_vars)
        return env_vars
    
    def _load_hosts_config(self) -> Dict[str, Any]:
        """Load hosts configuration from file
        
        Format:
        {
            "env_name": {
                "hosts": ["host1", "host2"],
                "mode": "docker" | "basilica"  # optional, defaults to docker
            },
            "default": {
                "hosts": ["localhost"],
                "mode": "docker"
            }
        }
        """
        # Check for config file in multiple locations
        config_paths = [
            Path(os.getenv("AFFINETES_HOSTS_CONFIG", "")),
            Path.cwd() / "affinetes_hosts.json",
            Path.home() / ".affine" / "hosts.json",
            Path("/etc/affine/hosts.json"),
        ]
        
        for config_path in config_paths:
            if config_path.exists() and config_path.is_file():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    logger.debug(f"Loaded hosts config from: {config_path}")
                    return config
                except Exception as e:
                    logger.warning(f"Failed to load hosts config from {config_path}: {e}")
        
        return {}
    
    def _get_hosts_and_mode(self) -> tuple[List[str], str]:
        """Get hosts and execution mode for this environment
        
        Returns:
            (hosts, mode): hosts list and execution mode ('docker' or 'basilica')
        """
        # Try config file first
        config = self._load_hosts_config()
        
        if config:
            # Check for environment-specific config
            if self.env_name in config:
                env_config = config[self.env_name]
                if isinstance(env_config, dict):
                    hosts = env_config.get("hosts", ["localhost"])
                    mode = env_config.get("mode", "docker")
                    logger.debug(f"Using config for {self.env_name}: hosts={hosts}, mode={mode}")
                    return hosts, mode
                elif isinstance(env_config, list):
                    # Backward compatibility: ["host1", "host2"]
                    logger.debug(f"Using config hosts for {self.env_name}: {env_config}")
                    return env_config, "docker"
            
            # Fall back to default config
            if "default" in config:
                default_config = config["default"]
                if isinstance(default_config, dict):
                    hosts = default_config.get("hosts", ["localhost"])
                    mode = default_config.get("mode", "docker")
                    logger.debug(f"Using default config for {self.env_name}: hosts={hosts}, mode={mode}")
                    return hosts, mode
                elif isinstance(default_config, list):
                    logger.debug(f"Using default hosts for {self.env_name}: {default_config}")
                    return default_config, "docker"
        
        # Fall back to environment variable (for backward compatibility)
        hosts_env = os.getenv("AFFINETES_HOSTS", "").strip()
        if hosts_env:
            hosts = [h.strip() for h in hosts_env.split(",") if h.strip()]
            if hosts:
                logger.debug(f"Using env var hosts for {self.env_name}: {hosts}")
                return hosts, "docker"
        
        return ["localhost"], "docker"
    
    def _load_environment(self) -> Any:
        """Load or get cached environment instance
        
        Mode selection priority:
        1. mode parameter passed to __init__
        2. mode from affinetes_hosts.json
        3. AFFINETES_MODE environment variable
        4. default to 'docker'
        """
        with _ENV_LOCK:
            if self.env_name in _ENV_CACHE:
                cached = _ENV_CACHE[self.env_name]
                if cached.is_ready():
                    logger.debug(f"Reusing cached environment: {self.env_name}")
                    return cached
                del _ENV_CACHE[self.env_name]
            
            # Determine execution mode
            hosts, config_mode = self._get_hosts_and_mode()
            
            # Priority: parameter > config > env var > default
            if self._mode_override:
                mode = self._mode_override
                logger.info(f"Using mode from parameter: {mode}")
            elif config_mode:
                mode = config_mode
                logger.info(f"Using mode from config: {mode}")
            else:
                mode = os.getenv("AFFINETES_MODE", "docker")
                logger.info(f"Using mode from env/default: {mode}")
            
            # Validate mode
            if mode not in ["docker", "basilica"]:
                raise ValueError(f"Invalid mode: {mode}. Must be 'docker' or 'basilica'")
            
            # Load environment
            logger.info(f"Loading environment: {self.env_name} (image={self.docker_image}, mode={mode}, hosts={hosts or 'local'}, mem_limit={self.config.mem_limit})")

            # Convert memory format for the selected mode
            mem_limit = convert_memory_format(self.config.mem_limit, mode)

            # Build load_env kwargs based on mode
            load_kwargs = {
                "image": self.docker_image,
                "mode": mode,
                "env_vars": self._get_env_vars(),
                "mem_limit": mem_limit,
                "pull": True,
            }
            
            if mode == "docker":
                # Docker mode specific parameters
                load_kwargs.update({
                    "replicas": len(hosts),
                    "hosts": hosts,
                    "container_name": self.env_name.replace(":", "-"),
                    "force_recreate": True,
                })
            elif mode == "basilica":
                # Basilica mode specific parameters
                ttl_buffer = self.config.proxy_timeout
                cpu_limit = self.config.cpu_limit or "2000m"
                
                load_kwargs.update({
                    "cpu_limit": cpu_limit,
                    "ttl_buffer": ttl_buffer,
                })
            
            # Add volumes if configured (both modes)
            if self.config.volumes:
                load_kwargs["volumes"] = self.config.volumes
            
            env = af_env.load_env(**load_kwargs)
            
            _ENV_CACHE[self.env_name] = env
            logger.debug(f"Cached environment: {self.env_name} (mode={mode})")
            return env
    
    def _generate_seed(self, task_id: int) -> int:
        """Generate deterministic seed"""
        seed_string = f"{self.env_name}:{task_id}"
        hash_bytes = hashlib.sha256(seed_string.encode()).digest()[:8]
        return int.from_bytes(hash_bytes, byteorder='big') % (2**32)
    
    def _prepare_eval_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Prepare evaluation kwargs based on environment configuration"""
        if "task_id" not in kwargs:
            raise ValueError("task_id is required for evaluation")
        
        # Generate seed if not provided
        if "seed" not in kwargs:
            kwargs["seed"] = self._generate_seed(kwargs["task_id"])
        
        # Merge eval_params from config (user-provided kwargs take precedence)
        for key, value in self.config.eval_params.items():
            kwargs.setdefault(key, value)
        
        return kwargs
    
    async def _evaluate_single(self, miner: Optional["Miner"], **kwargs) -> Result:
        """Evaluate single miner"""
        start = time.monotonic()
        kwargs = self._prepare_eval_kwargs(**kwargs)
        
        # Build payload with miner info
        payload = kwargs.copy()
        if miner and hasattr(miner, 'slug') and miner.slug:
            payload.update({
                "model": miner.model,
                "base_url": f"https://{miner.slug}.chutes.ai/v1"
            })
        
        result = await self._env.evaluate(_timeout=self.config.proxy_timeout, **payload)
        
        return self._build_result(result, miner, payload, start)
    
    def _build_result(self, result: Dict[str, Any], miner: Optional["Miner"], 
                     payload: Dict[str, Any], start_time: float) -> Result:
        """Build Result object from evaluation result"""
        extra = result.get("extra", {}).copy()
        extra["image"] = self.docker_image
        extra["request"] = payload.copy()
        
        return Result(
            miner_hotkey=miner.hotkey if miner else "",
            model_revision=miner.revision if miner else "",
            env=self.env_name,
            score=float(result.get("score", 0.0)),
            latency_seconds=time.monotonic() - start_time,
            success=bool(result.get("success", False)),
            error=result.get("error"),
            task_id=payload.get("task_id"),
            extra=extra,
            timestamp=time.time()
        )
    
    async def evaluate(self, miner: Optional[Union["Miner", Dict[str, "Miner"]]] = None, 
                      **kwargs) -> Union[Result, Dict[str, Result]]:
        """Evaluate miner(s)"""
        if isinstance(miner, dict):
            results = {}
            for key, m in miner.items():
                if self._validate_miner(m):
                    results[key] = await self._evaluate_single(m, **kwargs)
                else:
                    logger.warning(f"Skipping invalid miner: {key}")
            return results
        else:
            return await self._evaluate_single(miner, **kwargs)
    
    async def evaluate_batch(self, miners: List[Union["Miner", Dict[str, Any]]], 
                            **kwargs) -> List[Result]:
        """Evaluate multiple miners in parallel"""
        tasks = [self.evaluate(m, **kwargs) for m in miners]
        return await asyncio.gather(*tasks)
    
    @staticmethod
    def _validate_miner(miner: Any) -> bool:
        """Validate miner object"""
        return (hasattr(miner, "model") and hasattr(miner, "slug") and 
                miner.model and miner.slug)


# ========================= Factory Functions =========================

def create_environment(env_name: str, mode: Optional[str] = None) -> SDKEnvironment:
    """Create environment by name
    
    Args:
        env_name: Environment name
        mode: Execution mode ('docker' or 'basilica'). If not specified, uses config/default.
    """
    return SDKEnvironment(env_name, mode=mode)


def list_available_environments() -> Dict[str, List[str]]:
    """List all available environments grouped by type"""
    result = {}
    for name, config in ENV_CONFIGS.items():
        env_type = config.env_type
        result.setdefault(env_type, []).append(name)
    
    for env_type in result:
        result[env_type].sort()
    
    return result


def cleanup_all_environments():
    """Clean up all cached environments"""
    with _ENV_LOCK:
        logger.info("Cleaning up all cached environments")
        for name, env in list(_ENV_CACHE.items()):
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    loop.run_until_complete(env.cleanup())
                logger.debug(f"Cleaned up environment: {name}")
            except Exception as e:
                logger.warning(f"Error cleaning up environment {name}: {e}")
        
        _ENV_CACHE.clear()


# ========================= Backward Compatibility Aliases =========================

# Factory functions for backward compatibility
SAT_factory = lambda mode=None: create_environment("sat", mode=mode)
ABD_factory = lambda mode=None: create_environment("abd", mode=mode)  # Points to abd-v2
DED_factory = lambda mode=None: create_environment("ded", mode=mode)  # Points to ded-v2
DED_V2_factory = lambda mode=None: create_environment("ded-v2", mode=mode)
ABD_V2_factory = lambda mode=None: create_environment("abd-v2", mode=mode)
CDE_factory = lambda mode=None: create_environment("cde", mode=mode)
LGC_factory = lambda mode=None: create_environment("lgc", mode=mode)
LGC_V2_factory = lambda mode=None: create_environment("lgc-v2", mode=mode)
GAME_factory = lambda mode=None: create_environment("game", mode=mode)
SWE_PRO_factory = lambda mode=None: create_environment("swe-pro", mode=mode)
SWE_SYNTH_factory = lambda mode=None: create_environment("swe-synth", mode=mode)
PRINT_factory = lambda mode=None: create_environment("print", mode=mode)
ARC_GEN_factory = lambda mode=None: create_environment("arc-gen", mode=mode)

# Legacy class aliases
SAT = SAT_factory
ABD = ABD_factory
DED = DED_factory
DED_V2 = DED_V2_factory
ABD_V2 = ABD_V2_factory
CDE = CDE_factory
LGC = LGC_factory
LGC_V2 = LGC_V2_factory
GAME = GAME_factory
PRINT = PRINT_factory

# SWE-bench factories
SWE_PRO = SWE_PRO_factory
SWE_SYNTH = SWE_SYNTH_factory

# ARC-GEN factory
ARC_GEN = ARC_GEN_factory
