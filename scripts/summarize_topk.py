# scripts/summarize_topk.py

import json
from pathlib import Path

IN1 = Path("results/hf_batch_topk1.json")
IN20 = Path("results/hf_batch_topk20.json")
OUT = Path("results/topk_summary.json")

d1 = json.loads(IN1.read_text())
d20 = json.loads(IN20.read_text())

summary = {
    "top_k=1": d1["summary"],
    "top_k=20": d20["summary"],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(summary, indent=2))
print("[OK] wrote", OUT)
