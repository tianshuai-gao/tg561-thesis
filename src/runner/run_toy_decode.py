# runner/run_toy_decode.py

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.constraints.sqlguard import SQLGuardConstraint, SQLGuardConfig
from src.constraints.base import StepResult


@dataclass
class ToyReport:
    name: str
    n_steps: int
    ok_steps: int
    rejected_steps: int
    avg_step_ms: float
    reasons: Dict[str, int]


def run_sqlguard(seq: List[str], dialect: str) -> ToyReport:
    c = SQLGuardConstraint(SQLGuardConfig(dialect=dialect))
    c.reset()

    ok_steps = 0
    rejected = 0
    reasons: Dict[str, int] = {}

    t0 = time.time()
    for tok in seq:
        r: StepResult = c.step(tok)
        reason = (r.info or {}).get("reason", "none")
        reasons[reason] = reasons.get(reason, 0) + 1
        if r.ok:
            ok_steps += 1
        else:
            rejected += 1
            break
    t1 = time.time()

    n = len(seq)
    avg_ms = (t1 - t0) * 1000.0 / max(n, 1)
    return ToyReport(
        name="sqlguard",
        n_steps=n,
        ok_steps=ok_steps,
        rejected_steps=rejected,
        avg_step_ms=avg_ms,
        reasons=reasons,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/toy_sqlguard.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dialect = cfg.get("dialect", "sqlite")

    # Two toy sequences: one valid, one invalid
    good = ["SELECT ", "* ", "FROM ", "author", ";"]
    bad = ["SELECT ", "FROM ", ";"]

    rep_good = run_sqlguard(good, dialect)
    rep_bad = run_sqlguard(bad, dialect)

    out = {
        "engine": "sqlguard",
        "dialect": dialect,
        "good": rep_good.__dict__,
        "bad": rep_bad.__dict__,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()

