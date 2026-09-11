import sys
import io
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

df = pd.read_csv("data/twcs.csv")

candidates = ["SpotifyCares", "XboxSupport", "AmericanAir"]

for brand in candidates:
    brand_replies = df[df["author_id"] == brand]
    print(f"\n=== {brand} ===")
    print(f"total replies: {len(brand_replies)}")

    sample = brand_replies.sample(5, random_state=42)
    for _, row in sample.iterrows():
        inbound_id = row["in_response_to_tweet_id"]
        inbound = df[df["tweet_id"] == inbound_id]
        if len(inbound):
            print("  IN :", inbound.iloc[0]["text"][:140].replace("\n", " "))
        print("  OUT:", row["text"][:140].replace("\n", " "))
        print("  ---")
