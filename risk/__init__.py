"""Risk engine sub-package.

Provides cardiovascular risk estimators with explicit transparency about
model used, inputs used, and missing inputs.

IMPORTANT: These are *simplified, educational* implementations derived from
published equations. They are NOT regulatory-cleared software. The frontend
MUST surface this to users.
"""

from .qrisk3 import qrisk3
from .score2 import score2
from .who_ish import who_ish
from .router import route_and_run, list_models

__all__ = ["qrisk3", "score2", "who_ish", "route_and_run", "list_models"]
