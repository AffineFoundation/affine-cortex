"""Teacher subsystem.

Generates teacher-model rollouts with logprobs for the DISTILL evaluation
path. Two pieces:

  - :class:`TeacherWorker` (``worker.py``) picks random task_ids from
    ``CORPUS-EVAL`` (or other ``TEACHER_ENVS``), runs the configured
    teacher model with ``collect_logprobs=True``, and writes results to
    the **private** R2 bucket under ``pending/{ENV}/{epoch_ms}.json``.

  - :class:`TeacherMover` (``mover.py``) periodically promotes a random
    subset of pending rollouts to the **public** R2 bucket so the
    DISTILL environment container can read them during scoring. Stays
    paused while ``system_config.environments.DISTILL.enabled_for_sampling``
    is false.

Both run together inside ``af servers teacher``; the public DISTILL
scoring loop is independent and unchanged.
"""

from .mover import TeacherMover
from .worker import TeacherWorker

__all__ = ["TeacherWorker", "TeacherMover"]
