# src/runner/run_hf_batch.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import subprocess


def run_one(model: str, prompt: str, max_new_tokens: int, constraint: str, top_k: int, out_path: Path) -> Dict[str, Any]:
    cmd = [
        "python",
        "-m",
        "src.runner.run_hf_generate",
        "--model",
        model,
        "--prompt",
        prompt,
        "--max_new_tokens",
        str(max_new_tokens),
        "--constraint",
        constraint,
        "--top_k",
        str(top_k),
        "--out",
        str(out_path),
    ]
    subprocess.check_call(cmd)
    return json.loads(out_path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--prompts", required=True, help="path to txt file, one prompt per line")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, default=1)
    args = ap.parse_args()

    prompts = [ln.strip() for ln in Path(args.prompts).read_text().splitlines() if ln.strip()]
    out_root = Path(args.out)
    out_root.parent.mkdir(parents=True, exist_ok=True)

    per: List[Dict[str, Any]] = []
    for i, p in enumerate(prompts):
        pdir = out_root.parent / "runs"
        pdir.mkdir(parents=True, exist_ok=True)

        a = run_one(args.model, p, args.max_new_tokens, "none", args.top_k, pdir / f"{i:03d}_none.json")
        b = run_one(args.model, p, args.max_new_tokens, "sqlguard", args.top_k, pdir / f"{i:03d}_sqlguard.json")

        per.append({"i": i, "prompt": p, "none": a, "sqlguard": b})

    def summarize(key: str) -> Dict[str, Any]:
        xs = [r[key] for r in per]
        return {
            "n": len(xs),
            "avg_accepted": mean(x["accepted_tokens"] for x in xs),
            "avg_rejected": mean(x["rejected_tokens"] for x in xs),
            "avg_ms": mean(x["avg_token_ms"] for x in xs),
        }

    summary = {
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "summary": {
            "none": summarize("none"),
            "sqlguard": summarize("sqlguard"),
        },
        "per_prompt": per,
    }

    out_root.write_text(json.dumps(summary, indent=2))
    print(f"[OK] wrote {out_root}")


if __name__ == "__main__":
    main()

