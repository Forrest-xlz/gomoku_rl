"""PPOReview: CPU-rollout-buffer PPO for post-Elo mixed-bucket review.

This module deliberately keeps the normal ``ppo.py`` untouched.  It builds a
separate policy class from the already existing ``ppo_cpu_buffer.py`` source, but
renames the class to ``PPOReview`` before execution so it does not collide with
the original ``PPO`` registration.

Use it through ``get_policy('ppo_review', ...)``.  The runner creates temporary
PPOReview policies only for review; normal self-play / bucket training can keep
using the original ``algo: ppo``.
"""

from __future__ import annotations

import re
from pathlib import Path

from .base import Policy

_SOURCE_PATH = Path(__file__).with_name("ppo_cpu_buffer.py")
if not _SOURCE_PATH.is_file():
    raise FileNotFoundError(
        "ppo_review.py requires gomoku_rl/policy/ppo_cpu_buffer.py to exist. "
        "Keep the CPU-buffer PPO file from the psro-bucket-review branch."
    )

_source = _SOURCE_PATH.read_text(encoding="utf-8")
_source, _count = re.subn(
    r"class\s+PPO\s*\(\s*Policy\s*\)\s*:",
    "class PPOReview(Policy):",
    _source,
    count=1,
)
if _count != 1:
    raise RuntimeError("Could not find exactly one `class PPO(Policy):` in ppo_cpu_buffer.py")

# Execute the CPU-buffer PPO implementation in this module's namespace after the
# class rename. Relative imports inside ppo_cpu_buffer.py still resolve because
# this module has the same package as gomoku_rl.policy.
exec(compile(_source, str(_SOURCE_PATH), "exec"), globals())

# Policy.__init_subclass__ registers PPOReview and pporeview automatically. Add
# an explicit underscore alias so cfg/runner code can call get_policy('ppo_review').
Policy.REGISTRY["ppo_review"] = PPOReview
Policy.REGISTRY["PPOReview"] = PPOReview

__all__ = ["PPOReview", "ManualPPOLoss"]
