"""
Evaluation harness. Consumes data/golden_labeled.csv (columns: customer_text,
actual_brand_reply, your_intent, your_escalate_decision) and scores the
pipeline against it, plus two baselines.

Baselines:
  - trivial: always predict the majority-class intent, always AUTO_HANDLE
  - simple: keyword-rule intent classifier + keyword-rule escalation
"""
import sys
import csv
import json
from collections import Counter

sys.path.insert(0, "src")
from pipeline import classify_intent, draft_reply, decide_escalation, INTENTS
from retrieval import Retriever
from llm import chat

KEYWORD_RULES = {
    "hardware_issue": ["won't turn on", "red ring", "power supply", "controller connect", "won't power"],
    "game_app_crash": ["freeze", "crash", "won't launch", "wont launch", "uninstall"],
    "account_signin": ["sign in", "signin", "log in", "login", "password"],
    "billing_subscription": ["charged", "subscription", "refund", "renewal", "cancel"],
    "purchase_store": ["buy", "purchase", "payment", "can't process", "store"],
    "enforcement_ban": ["ban", "suspend", "enforcement", "appeal"],
    "achievements_rewards": ["achievement", "reward", "unlock"],
    "network_service": ["wifi", "connect", "lag", "down", "outage", "live service"],
    "howto_general": ["how do i", "how to", "can i", "is there a way"],
}


def keyword_classify(text):
    t = text.lower()
    for intent, kws in KEYWORD_RULES.items():
        if any(kw in t for kw in kws):
            return intent
    return "noise_other"


def keyword_escalate(intent, text):
    t = text.lower()
    if intent in ("enforcement_ban", "billing_subscription"):
        return "ESCALATE"
    if any(w in t for w in ["fuck", "scam", "lawsuit", "sue", "awful", "worst"]):
        return "ESCALATE"
    return "AUTO_HANDLE"


JUDGE_PROMPT = """Rate this drafted customer support reply on a 1-5 scale for each dimension.

Customer message: "{text}"
Drafted reply: "{reply}"
Real historical reply from the brand (reference, not necessarily perfect): "{reference}"

Dimensions:
- grounded: does it match the style/approach the brand actually uses?
- helpful: does it give an actionable next step or real answer?
- tone: professional, on-brand?

Respond ONLY as JSON: {{"grounded": <1-5>, "helpful": <1-5>, "tone": <1-5>}}"""


def llm_judge(text, reply, reference):
    result = chat([{"role": "user", "content": JUDGE_PROMPT.format(text=text, reply=reply, reference=reference)}], max_tokens=60)
    try:
        return json.loads(result.strip().strip("`").replace("json\n", ""))
    except Exception:
        return {"grounded": None, "helpful": None, "tone": None}


def load_golden(path="data/golden_labeled.csv"):
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["your_intent"].strip():
                rows.append(row)
    return rows


def run_eval(golden_path="data/golden_labeled.csv", limit=None, resume=False):
    rows = load_golden(golden_path)
    if limit:
        rows = rows[:limit]

    retriever = Retriever()
    majority_intent = Counter(r["your_intent"] for r in rows).most_common(1)[0][0]

    results = {"pipeline": [], "trivial": [], "simple": []}
    start_idx = 0
    if resume:
        try:
            with open("data/eval_raw_results.partial.json", encoding="utf-8") as f:
                results = json.load(f)
            start_idx = len(results["pipeline"])
            print(f"Resuming from checkpoint at row {start_idx}")
        except FileNotFoundError:
            pass

    for idx, row in enumerate(rows):
        if idx < start_idx:
            continue
        text = row["customer_text"]
        true_intent = row["your_intent"]
        true_escalate = row["your_escalate_decision"].strip().upper()
        prior_context = json.loads(row["prior_context"]) if row.get("prior_context") else None

        # trivial baseline
        results["trivial"].append({
            "true_intent": true_intent, "pred_intent": majority_intent,
            "true_escalate": true_escalate, "pred_escalate": "AUTO_HANDLE",
        })

        # simple keyword baseline
        kw_intent = keyword_classify(text)
        kw_escalate = keyword_escalate(kw_intent, text)
        results["simple"].append({
            "true_intent": true_intent, "pred_intent": kw_intent,
            "true_escalate": true_escalate, "pred_escalate": kw_escalate,
        })

        # full pipeline
        pred_intent = classify_intent(text, prior_context)
        reply, _ = draft_reply(text, pred_intent, retriever)
        pred_escalate, reason = decide_escalation(text, pred_intent, reply)
        judge = llm_judge(text, reply, row["actual_brand_reply"])
        results["pipeline"].append({
            "true_intent": true_intent, "pred_intent": pred_intent,
            "true_escalate": true_escalate, "pred_escalate": pred_escalate,
            "reply": reply, "judge": judge,
        })

        if (idx + 1) % 10 == 0:
            print(f"{idx + 1}/{len(rows)} done")
            with open("data/eval_raw_results.partial.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def score(results_for_system):
    n = len(results_for_system)
    intent_correct = sum(1 for r in results_for_system if r["pred_intent"] == r["true_intent"])
    esc_correct = sum(1 for r in results_for_system if r["pred_escalate"] == r["true_escalate"])

    tp = sum(1 for r in results_for_system if r["pred_escalate"] == "ESCALATE" and r["true_escalate"] == "ESCALATE")
    fp = sum(1 for r in results_for_system if r["pred_escalate"] == "ESCALATE" and r["true_escalate"] == "AUTO_HANDLE")
    fn = sum(1 for r in results_for_system if r["pred_escalate"] == "AUTO_HANDLE" and r["true_escalate"] == "ESCALATE")
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    out = {
        "n": n,
        "intent_accuracy": round(intent_correct / n, 3),
        "escalation_accuracy": round(esc_correct / n, 3),
        "escalation_precision": round(precision, 3),
        "escalation_recall": round(recall, 3),
        "escalation_f1": round(f1, 3),
    }

    judges = [r["judge"] for r in results_for_system if "judge" in r and r["judge"]["grounded"] is not None]
    if judges:
        out["avg_grounded"] = round(sum(j["grounded"] for j in judges) / len(judges), 2)
        out["avg_helpful"] = round(sum(j["helpful"] for j in judges) / len(judges), 2)
        out["avg_tone"] = round(sum(j["tone"] for j in judges) / len(judges), 2)

    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    resume = "--resume" in args
    args = [a for a in args if a != "--resume"]
    limit = int(args[0]) if args else None
    results = run_eval(limit=limit, resume=resume)
    report = {name: score(rows) for name, rows in results.items()}
    print(json.dumps(report, indent=2))

    with open("data/eval_raw_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open("data/eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
