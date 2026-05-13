"""``af servers teacher`` entry point.

Spawns the teacher worker + mover as a single asyncio service so
DISTILL has a steady supply of fresh rollouts in the public R2 bucket.
"""

from affine.src.teacher.worker import main as _worker_main


main = _worker_main


if __name__ == "__main__":
    main()
