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
    top_k: int
    accepted_tokens: int
    rejected_tokens: int
    dead_end: bool
    total_ms: float
    avg_token_ms: float
    reject_reasons: Dict[str, int]
    output_text: str


def _make_guard() -> SQLGuardConstraint:
    g = SQLGuardConstraint(SQLGuardConfig(dialect="sqlite"))
    g.reset()
    return g


def _check_with_guard(prefix_text: str, cand_text: str) -> tuple[bool, str]:
    """
    Pure function check:
    run guard from scratch on (prefix_text + cand_text).
    Return (ok, reason).
    """
    g = _make_guard()
    # replay accepted prefix
    for ch in prefix_text:
        g.step(ch)
    r = g.step(cand_text)
    reason = (r.info or {}).get("reason", "null")
    return bool(r.ok), str(reason)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--constraint", choices=["none", "sqlguard"], default="none")
    ap.add_argument("--top_k", type=int, default=1, help="top-k resample when constrained; 1=greedy")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)

    accepted = 0
    rejected = 0
    reasons: Dict[str, int] = {}
    dead_end = False

    # We keep track of the generated suffix text (after the prompt)
    suffix_text = ""

    t0 = time.time()
    out_ids = input_ids

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            logits = model(out_ids).logits[:, -1, :]  # [1, vocab]

            if args.constraint == "none":
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
                out_ids = torch.cat([out_ids, next_id], dim=1)
                accepted += 1
                if next_id.item() == tok.eos_token_id:
                    break
                continue

            # constrained: try up to top_k candidates
            k = max(1, int(args.top_k))
            topk = torch.topk(logits, k=k, dim=-1)
            cand_ids = topk.indices[0].tolist()

            picked: Optional[int] = None
            picked_text: Optional[str] = None

            for cid in cand_ids:
                cand_text = tok.decode([cid], skip_special_tokens=False)
                ok, reason = _check_with_guard(suffix_text, cand_text)
                if ok:
                    picked = cid
                    picked_text = cand_text
                    reasons[reason] = reasons.get(reason, 0) + 1
                    break
                else:
                    rejected += 1
                    reasons[reason] = reasons.get(reason, 0) + 1

            if picked is None:
                dead_end = True
                break

            next_id = torch.tensor([[picked]], device=device)
            out_ids = torch.cat([out_ids, next_id], dim=1)
            accepted += 1
            suffix_text += picked_text  # advance state only on accept

            if picked == tok.eos_token_id:
                break

    t1 = time.time()
    total_ms = (t1 - t0) * 1000.0
    avg_token_ms = total_ms / max(accepted + rejected, 1)

    output_text = tok.decode(out_ids[0], skip_special_tokens=False)

    out = GenMetrics(
        model=args.model,
        device=device,
        max_new_tokens=args.max_new_tokens,
        constraint=args.constraint,
        top_k=int(args.top_k),
        accepted_tokens=accepted,
        rejected_tokens=rejected,
        dead_end=dead_end,
        total_ms=total_ms,
        avg_token_ms=avg_token_ms,
        reject_reasons=reasons,
        output_text=output_text,
    )

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(out), indent=2))
    print(f"[OK] wrote {p}")


if __name__ == "__main__":
    main()

