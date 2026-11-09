import json
from pathlib import Path

COLUMNS = ["recall@1", "recall@5", "recall@10", "mrr@10", "ms_per_query"]

for split in ["tune", "test"]:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in Path("results").glob(f"*_{split}.json")]
    if not rows:
        continue
    print(f"\n### {split} ({rows[0]['queries']} queries)\n")
    print("| config | " + " | ".join(COLUMNS) + " |")
    print("|---" * (len(COLUMNS) + 1) + "|")
    for r in sorted(rows, key=lambda r: r["mrr@10"], reverse=True):
        cells = [f"{r[c]:.0f}" if c == "ms_per_query" else f"{r[c]:.2f}" for c in COLUMNS]
        print(f"| {r['name']} | " + " | ".join(cells) + " |")
