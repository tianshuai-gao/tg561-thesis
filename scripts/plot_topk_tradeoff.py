# scripts/plot_topk_tradeoff.py

import json
from pathlib import Path

import matplotlib.pyplot as plt

INP = Path("results/topk_summary.json")
OUT_PNG = Path("plots/fig_topk_tradeoff.png")
OUT_PDF = Path("plots/fig_topk_tradeoff.pdf")

d = json.loads(INP.read_text())

topks = [1, 20]
engines = ["none", "sqlguard"]
metrics = ["avg_accepted", "avg_rejected", "avg_ms"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

for ax, engine in zip(axes, engines):
    for m in metrics:
        ys = []
        for k in topks:
            ys.append(d[f"top_k={k}"][engine][m])
        ax.plot(topks, ys, marker="o", label=m)
    ax.set_title(engine)
    ax.set_xlabel("top_k")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)

axes[0].set_ylabel("value")

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=200)
fig.savefig(OUT_PDF)
print("[OK] wrote", OUT_PNG, "and", OUT_PDF)