#!/usr/bin/env python3
"""Dispatch the autoscale provider wrapper selected by environment.

Docker Compose should only need one provider-wrapper service. The concrete GPU
platform is selected at runtime so deployments can switch between Targon and
Lium without adding one service per provider.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROVIDER_ENV = "AFFINE_GPU_AUTOSCALE_WRAPPER_PROVIDER"
DEFAULT_PROVIDER = "targon"
WRAPPER_BY_PROVIDER = {
    "lium": "lium_autoscale_wrapper.py",
    "targon": "targon_autoscale_wrapper.py",
}


def selected_provider() -> str:
    return (os.getenv(PROVIDER_ENV) or DEFAULT_PROVIDER).strip().lower()


def wrapper_script(provider: str) -> Path:
    script = WRAPPER_BY_PROVIDER.get(provider)
    if script is None:
        valid = ", ".join(sorted(WRAPPER_BY_PROVIDER))
        raise ValueError(f"{PROVIDER_ENV} must be one of: {valid}; got {provider!r}")
    return Path(__file__).resolve().with_name(script)


def main() -> None:
    provider = selected_provider()
    script = wrapper_script(provider)
    print(
        f"gpu-autoscale-wrapper: provider={provider} script={script}",
        flush=True,
    )
    os.execv(sys.executable, [sys.executable, str(script), *sys.argv[1:]])


if __name__ == "__main__":
    main()
