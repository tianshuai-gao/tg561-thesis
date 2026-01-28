# runner/run.py

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class RunResult:
    name: str
    status: str
    metrics: Dict[str, Any]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--out", default="results/results.json", help="Output json path")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(cfg_path.read_text())

    # Phase 2.1: dry-run only (no heavy compute on Mac)
    result = RunResult(
        name=cfg.get("name", cfg_path.stem),
        status="dry_run_ok",
        metrics={
            "engine": cfg.get("engine"),
            "note": "dry-run on mac; real runs happen on CSD3",
            "config": cfg,
        },
    )

    out_path.write_text(json.dumps(result.__dict__, indent=2))
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
