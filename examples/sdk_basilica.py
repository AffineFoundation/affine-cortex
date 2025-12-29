#!/usr/bin/env python3
"""
Affine SDK Example - GAME Environment with Basilica Mode

This example demonstrates how to use the GAME environment with Basilica mode,
where each evaluation task runs in a temporary Kubernetes pod.

Mode Selection Priority:
1. Explicit mode parameter: af.GAME(mode="basilica")
2. affinetes_hosts.json configuration
3. AFFINETES_MODE environment variable
4. Default: docker mode
"""

import asyncio
import os
import sys
import json
from dotenv import load_dotenv

import affine as af

# Enable tracing
af.trace()

# Load environment variables
load_dotenv()


async def main():
    """Main example using GAME environment with basilica mode"""
    
    # Check required environment variable
    chutes_api_key = os.getenv("CHUTES_API_KEY")
    
    if not chutes_api_key:
        print("\n   ❌ CHUTES_API_KEY environment variable not set")
        print("   Please set: export CHUTES_API_KEY='your-key'")
        sys.exit(1)
    
    uid = 94
    miner = await af.miners(uid)
    if not miner:
        print("   ❌ Miner not found")
        print("   Please check if the miner is registered")
        sys.exit(1)

    game_env = af.GAME(mode="basilica")
    try:
        print("🚀 Starting evaluation...")
        evaluation = await game_env.evaluate(miner, task_id=388240510)
        
        print("\n✅ Evaluation completed!")
        print(json.dumps(evaluation[uid].dict(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__ == "__main__":
    asyncio.run(main())