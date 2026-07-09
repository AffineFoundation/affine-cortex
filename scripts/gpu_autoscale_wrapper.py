#!/usr/bin/env python3
"""Dispatch autoscale provider wrappers from one Compose service.

Docker Compose should only need one provider-wrapper service. By default this
launcher starts both provider listeners so Lium/Targon fallback keeps working
while operators still manage a single container. Set
``AFFINE_GPU_AUTOSCALE_WRAPPER_PROVIDER`` to ``lium`` or ``targon`` to run only
one provider for debugging or single-provider deployments.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


PROVIDER_ENV = "AFFINE_GPU_AUTOSCALE_WRAPPER_PROVIDER"
DEFAULT_PROVIDER = "all"
ALL_PROVIDERS = ("lium", "targon")
WRAPPER_BY_PROVIDER = {
    "lium": "lium_autoscale_wrapper.py",
    "targon": "targon_autoscale_wrapper.py",
}


def selected_providers() -> tuple[str, ...]:
    raw = (os.getenv(PROVIDER_ENV) or DEFAULT_PROVIDER).strip().lower()
    if raw in {"all", "*"}:
        return ALL_PROVIDERS
    providers = tuple(
        dict.fromkeys(part.strip() for part in raw.split(",") if part.strip())
    )
    if not providers:
        return ALL_PROVIDERS
    for provider in providers:
        if provider not in WRAPPER_BY_PROVIDER:
            valid = ", ".join(("all", *sorted(WRAPPER_BY_PROVIDER)))
            raise ValueError(
                f"{PROVIDER_ENV} must be one of: {valid}; got {provider!r}"
            )
    return providers


def wrapper_script(provider: str) -> Path:
    script = WRAPPER_BY_PROVIDER.get(provider)
    if script is None:
        valid = ", ".join(sorted(WRAPPER_BY_PROVIDER))
        raise ValueError(f"{PROVIDER_ENV} must be one of: {valid}; got {provider!r}")
    return Path(__file__).resolve().with_name(script)


def _terminate(children: list[subprocess.Popen]) -> None:
    for child in children:
        if child.poll() is None:
            child.terminate()
    deadline = time.time() + 10
    for child in children:
        remaining = max(0.0, deadline - time.time())
        try:
            child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            child.kill()


def run_many(providers: tuple[str, ...]) -> int:
    children: list[subprocess.Popen] = []

    def _handle_signal(signum, _frame):
        _terminate(children)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    for provider in providers:
        script = wrapper_script(provider)
        print(
            f"gpu-autoscale-wrapper: starting provider={provider} script={script}",
            flush=True,
        )
        children.append(subprocess.Popen([sys.executable, str(script), *sys.argv[1:]]))

    while True:
        for idx, child in enumerate(children):
            rc = child.poll()
            if rc is not None:
                provider = providers[idx]
                print(
                    f"gpu-autoscale-wrapper: provider={provider} exited rc={rc}; "
                    "stopping sibling wrappers",
                    flush=True,
                )
                _terminate(children)
                return int(rc or 0)
        time.sleep(1)


def main() -> None:
    providers = selected_providers()
    if len(providers) > 1:
        raise SystemExit(run_many(providers))

    provider = providers[0]
    script = wrapper_script(provider)
    print(
        f"gpu-autoscale-wrapper: provider={provider} script={script}",
        flush=True,
    )
    os.execv(sys.executable, [sys.executable, str(script), *sys.argv[1:]])


if __name__ == "__main__":
    main()
