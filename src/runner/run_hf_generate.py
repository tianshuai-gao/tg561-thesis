# src/runner/run_hf_generate.py

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.constraints.sqlguard import SQLGuardConstraint, SQLGuardConfig


@dataclass
class GenMetrics:
    model: str
    device: str
    max_new_tokens: int
    constraint: str
    accepted_tokens: int
    rejected_tokens: int
    total_ms: float
    avg_token_ms: float
    reject_reasons: Dict[str, int]
    output_text: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--constraint", choices=["none", "sqlguard"], default="none")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    guard: Optional[SQLGuardConstraint] = None
    if args.constraint == "sqlguard":
        guard = SQLGuardConstraint(SQLGuardConfig(dialect="sqlite"))
        guard.reset()

    input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)

    accepted = 0
    rejected = 0
    reasons: Dict[str, int] = {}

    t0 = time.time()
    out_ids = input_ids
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            logits = model(out_ids).logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            next_text = tok.decode(next_id[0], skip_special_tokens=False)

            if guard is not None:
                r = guard.step(next_text)
                reason = (r.info or {}).get("reason", "none")
                reasons[reason] = reasons.get(reason, 0) + 1
                if not r.ok:
                    rejected += 1
                    break

            out_ids = torch.cat([out_ids, next_id], dim=1)
            accepted += 1

            if next_id.item() == tok.eos_token_id:
                break
    t1 = time.time()

    gen_text = tok.decode(out_ids[0], skip_special_tokens=False)
    total_ms = (t1 - t0) * 1000.0
    avg_ms = total_ms / max(accepted + rejected, 1)

    m = GenMetrics(
        model=args.model,
        device=device,
        max_new_tokens=args.max_new_tokens,
        constraint=args.constraint,
        accepted_tokens=accepted,
        rejected_tokens=rejected,
        total_ms=total_ms,
        avg_token_ms=avg_ms,
        reject_reasons=reasons,
        output_text=gen_text,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(m), indent=2))
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()

