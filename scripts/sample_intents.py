import sys
import io
import json
import random

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

threads = []
with open("data/xbox_threads.jsonl", encoding="utf-8") as f:
    for line in f:
        threads.append(json.loads(line))

random.seed(42)
sample = random.sample(threads, 80)

for i, t in enumerate(sample):
    print(f"{i+1}. {t['customer_text'][:180]}")
