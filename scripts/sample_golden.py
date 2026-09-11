import sys
import io
import json
import random

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

threads = []
with open("data/xbox_threads.jsonl", encoding="utf-8") as f:
    for line in f:
        threads.append(json.loads(line))

random.seed(7)
sample = random.sample(threads, 200)

with open("data/golden_unlabeled.jsonl", "w", encoding="utf-8") as f:
    for t in sample:
        f.write(json.dumps({
            "customer_text": t["customer_text"],
            "actual_brand_reply": t["brand_reply_text"],
        }, ensure_ascii=False) + "\n")

print(f"Sampled {len(sample)} messages -> data/golden_unlabeled.jsonl")
