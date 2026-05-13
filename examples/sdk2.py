"""Affine SDK example — evaluate any model identifier directly.

Skips the on-chain miner lookup from ``sdk.py`` — useful when you want
to benchmark an arbitrary HuggingFace model against an Affine
environment without committing it on chain first.
"""

import asyncio
import json

from dotenv import load_dotenv

import affine as af


af.trace()
load_dotenv()


# OpenAI-compatible inference endpoint for the model below. Bring up
# vLLM / sglang / ollama / a managed gateway — anything that speaks
# /v1/chat/completions.
BASE_URL = "http://localhost:8000/v1"
MODEL = "deepseek-ai/DeepSeek-V3"


async def main() -> None:
    ded = af.DED()
    result = await ded.evaluate(model=MODEL, base_url=BASE_URL, task_id=20100)
    print("DED:", json.dumps(result.dict(), indent=2, ensure_ascii=False))

    abd = af.ABD()
    result = await abd.evaluate(model=MODEL, base_url=BASE_URL, task_id=20200)
    print("ABD:", json.dumps(result.dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
