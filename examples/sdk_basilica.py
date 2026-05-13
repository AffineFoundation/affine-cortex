"""Affine SDK example — GAME environment in Basilica mode.

Basilica runs each evaluation task in a temporary cloud pod instead of a
local Docker container. Useful for environments that need more RAM/CPU
than your laptop has. Requires ``BASILICA_API_TOKEN``.

Mode selection priority:
  1. Explicit ``mode="basilica"`` argument (used here)
  2. ``affinetes_hosts.json`` config
  3. ``AFFINETES_MODE`` env var
  4. Default: ``docker``
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv

import affine as af


af.trace()
load_dotenv()


# OpenAI-compatible base URL serving the model below.
BASE_URL = os.getenv("AFFINE_SDK_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("AFFINE_SDK_MODEL", "deepseek-ai/DeepSeek-V3")


async def main() -> None:
    if not os.getenv("BASILICA_API_TOKEN"):
        print("BASILICA_API_TOKEN not set — Basilica mode requires it.")
        sys.exit(1)

    game = af.GAME(mode="basilica")
    print("Starting GAME evaluation on Basilica...")
    try:
        result = await game.evaluate(
            model=MODEL, base_url=BASE_URL, task_id=388_240_510,
        )
        print(json.dumps(result.dict(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Evaluation failed: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
