# constraints/base.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass
class StepResult:
    ok: bool
    info: Optional[Dict[str, Any]] = None


class Constraint(Protocol):
    """
    Unified constraint API used across PICARD / XGrammar / Hybrid.
    """
    def reset(self, **kwargs: Any) -> None:
        ...

    def step(self, text: str) -> StepResult:
        """
        Consume a text chunk (token string or string fragment) and update state.
        """
        ...
