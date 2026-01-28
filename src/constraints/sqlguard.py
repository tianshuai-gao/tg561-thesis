# constraints/sqlguard.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import sqlglot

from .base import StepResult


@dataclass
class SQLGuardConfig:
    dialect: str = "sqlite"
    allow_before_select: bool = True
    treat_incomplete_as_ok: bool = True


class SQLGuardConstraint:
    """
    Plan B "PICARD-like" semantic guard:
    - uses a real SQL parser (sqlglot) to validate (partial) SQL strings.
    - reproduces the KEY mechanism: reject invalid candidates during decoding.

    NOTE: This is not the original PICARD thrift+haskell service.
    """
    def __init__(self, cfg: SQLGuardConfig = SQLGuardConfig()):
        self.cfg = cfg
        self._buf = ""

    def reset(self, **kwargs: Any) -> None:
        self._buf = ""

    def step(self, text: str) -> StepResult:
        self._buf += text
        ok, reason = self._check(self._buf)
        return StepResult(ok=ok, info={"reason": reason, "sql": self._buf})

    def _check(self, s: str) -> Tuple[bool, Optional[str]]:
        ss = s.strip()
        if not ss:
            return True, None

        # If statement looks "finished", be strict.
        finished = ss.endswith(";")

        if self.cfg.allow_before_select and "select" not in ss.lower():
            return True, "pre_select"

        try:
            sqlglot.parse_one(ss, read=self.cfg.dialect)
            return True, None
        except Exception as e:
            msg = str(e).lower()

            if not finished and self.cfg.treat_incomplete_as_ok:
                # allow incomplete mid-generation
                incomplete_signals = [
                    "expected",
                    "missing",
                    "end of input",
                    "eof",
                    "unterminated",
                ]
                if any(t in msg for t in incomplete_signals):
                    return True, "incomplete"

            return False, "parse_error"


