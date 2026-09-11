import sys
import io
import json
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BRAND = "XboxSupport"

df = pd.read_csv("data/twcs.csv")
df["in_response_to_tweet_id"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")
df["response_tweet_id_first"] = df["response_tweet_id"].astype(str).str.split(",").str[0]
df["response_tweet_id_first"] = pd.to_numeric(df["response_tweet_id_first"], errors="coerce")

by_id = df.set_index("tweet_id", drop=False)

brand_replies = df[(df["author_id"] == BRAND) & (df["in_response_to_tweet_id"].notna())]

threads = []
for _, reply in brand_replies.iterrows():
    inbound_id = reply["in_response_to_tweet_id"]
    if inbound_id not in by_id.index:
        continue
    inbound = by_id.loc[inbound_id]
    if isinstance(inbound, pd.DataFrame):
        inbound = inbound.iloc[0]
    if not inbound["inbound"]:
        continue

    # walk back further to capture prior context (customer's earlier turn if any)
    context = []
    cur = inbound
    depth = 0
    while depth < 3:
        prior_id = cur.get("in_response_to_tweet_id")
        if pd.isna(prior_id) or prior_id not in by_id.index:
            break
        prior = by_id.loc[prior_id]
        if isinstance(prior, pd.DataFrame):
            prior = prior.iloc[0]
        context.insert(0, {"author": prior["author_id"], "text": prior["text"], "inbound": bool(prior["inbound"])})
        cur = prior
        depth += 1

    threads.append({
        "reply_tweet_id": int(reply["tweet_id"]),
        "customer_tweet_id": int(inbound["tweet_id"]),
        "customer_text": inbound["text"],
        "brand_reply_text": reply["text"],
        "prior_context": context,
        "created_at": reply["created_at"],
    })

print(f"Total XboxSupport threads reconstructed: {len(threads)}")

with open("data/xbox_threads.jsonl", "w", encoding="utf-8") as f:
    for t in threads:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

print("Saved to data/xbox_threads.jsonl")
