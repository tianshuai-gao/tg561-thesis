# constraints/picard_wrapper.py

from __future__ import annotations
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from .base import StepResult


@dataclass
class PicardConfig:
    """
    Minimal config for wiring PICARD as an external dependency.

    We first implement a smoke-test checker. Later we will integrate
    into decoding loop with top-k / incremental / guards.
    """
    picard_root: Path


class PicardConstraint:
    """
    A minimal wrapper around PICARD that can validate SQL strings.

    Implementation strategy (phase-2):
    - call PICARD's own tooling/scripts inside external/picard
      (keeps our repo clean; avoids re-implementing the parser)
    - only used for smoke tests now

    Later (phase-3):
    - integrate incremental checking (top-k) into our decoding harness.
    """
    def __init__(self, cfg: PicardConfig):
        self.cfg = cfg
        self._buffer = ""

    def reset(self, **kwargs: Any) -> None:
        self._buffer = ""

    def step(self, text: str) -> StepResult:
        self._buffer += text
        ok = self._validate_sql(self._buffer)
        return StepResult(ok=ok, info={"sql": self._buffer})

    def _validate_sql(self, sql: str) -> bool:
        """
        Phase-2: simplest possible validity check.

        We call a tiny python helper that we will place under scripts/
        to avoid guessing PICARD internal APIs.

        Returns True if PICARD says parsable/valid under its mode.
        """
        helper = Path(__file__).resolve().parents[2] / "scripts" / "picard_check_sql.py"
        cmd = ["python", str(helper), "--sql", sql, "--picard_root", str(self.cfg.picard_root)]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return r.returncode == 0
