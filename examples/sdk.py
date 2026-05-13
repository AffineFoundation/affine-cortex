"""Affine SDK example — evaluate a model on one environment.

The SDK exposes ``af.miners(uid)`` to look up what model a UID committed
on chain, and per-environment factories (``af.DED()``, ``af.ABD()``, etc.)
that wrap the Affinetes evaluation containers.

In the queue-window flow, validators host inference per-window via
Targon, so there's no shared per-miner endpoint anymore. To evaluate
locally you bring up your own OpenAI-compatible inference server
(vLLM / sglang) for the model and pass its URL as ``base_url``.
"""

import asyncio
import json
import sys

from dotenv import load_dotenv

import affine as af


af.trace()
load_dotenv()


# Where your local model server is listening (vLLM / sglang / ollama-with-
# openai-shim / anything OpenAI-compatible). Adjust to taste.
BASE_URL = "http://localhost:8000/v1"


async def main() -> None:
    # 1) Read a miner's on-chain commitment to learn which model they
    #    submitted. ``af.miners()`` returns a slim record — just
    #    (uid, hotkey, model, revision, block).
    uid = 243
    miners = await af.miners(uid)
    if not miners or uid not in miners:
        print(f"Miner uid={uid} not found on chain")
        sys.exit(1)
    miner = miners[uid]
    print(f"Miner uid={miner.uid} model={miner.model}@{(miner.revision or '')[:8]}")

    # 2) Evaluate that model on DED-V2 / ABD-V2. The SDK never connects
    #    to the validator's inference; you point at your own server.
    ded = af.DED()
    result = await ded.evaluate(model=miner.model, base_url=BASE_URL, task_id=20100)
    print("DED:", json.dumps(result.dict(), indent=2, ensure_ascii=False))

    abd = af.ABD()
    result = await abd.evaluate(model=miner.model, base_url=BASE_URL, task_id=20200)
    print("ABD:", json.dumps(result.dict(), indent=2, ensure_ascii=False))

    # 3) List the environments the SDK knows about.
    print("Available environments:")
    for env_type, names in af.tasks.list_available_environments().items():
        print(f"  {env_type}:")
        for name in names:
            print(f"    - {name}")


if __name__ == "__main__":
    asyncio.run(main())
