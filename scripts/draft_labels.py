import sys
import io
import json
import csv

sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pipeline import classify_intent, INTENTS

rows = []
with open("data/golden_unlabeled.jsonl", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

with open("data/golden_draft.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["customer_text", "actual_brand_reply", "draft_intent", "your_intent", "your_escalate_decision", "notes"])
    for i, r in enumerate(rows):
        intent = classify_intent(r["customer_text"])
        writer.writerow([r["customer_text"], r["actual_brand_reply"], intent, "", "", ""])
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{len(rows)} labeled")

print("Saved data/golden_draft.csv")
