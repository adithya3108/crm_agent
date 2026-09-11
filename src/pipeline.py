import json
import sys
from llm import chat
from retrieval import Retriever

INTENTS = [
    "hardware_issue",
    "game_app_crash",
    "account_signin",
    "billing_subscription",
    "purchase_store",
    "enforcement_ban",
    "achievements_rewards",
    "network_service",
    "howto_general",
    "noise_other",
]

CLASSIFY_PROMPT = """You are classifying a customer support tweet sent to XboxSupport into exactly one intent.

Intents:
- hardware_issue: console/controller/accessory won't power on, red ring, physical connectivity/defects
- game_app_crash: game or app freezes, won't launch, crashes, install/uninstall bugs
- account_signin: can't sign in, password/account access problems
- billing_subscription: subscription charges, payment failures, unwanted renewals, refunds
- purchase_store: can't buy content, payment errors at point of sale, missing purchased/pre-ordered content
- enforcement_ban: suspensions, bans, harassment reports, appeal status
- achievements_rewards: achievements/rewards not unlocking, missing items
- network_service: Xbox Live outages, WiFi connectivity, lag, service status
- howto_general: feature questions, compatibility, how-to questions
- noise_other: thanks/praise/complaints with no actionable ask, off-topic, unparseable

{context_block}Customer message: "{text}"

Respond with ONLY the intent label, nothing else."""

ESCALATE_PROMPT = """Given this customer support case for Xbox, decide whether it should be AUTO_HANDLE (bot/standard flow can resolve it) or ESCALATE (needs a human specialist).

Intent: {intent}
Customer message: "{text}"
Drafted reply: "{reply}"

IMPORTANT: XboxSupport's normal, successful resolution pattern on Twitter is to ask the customer to DM their Gamertag or more details so a human agent can look into their account. This is NOT a sign of failure or vagueness — it is the standard auto-handleable first response for most technical/account cases. Do NOT escalate just because the reply asks for a DM or more info.

Escalate if ANY of:
- intent is enforcement_ban (suspension/ban appeals) or billing_subscription (money/refund disputes)
- the customer message expresses anger, threats, legal action, or explicit demand for compensation
- the case requires judgment calls a support script cannot make (e.g. disputing a company decision)
- the drafted reply gives NO concrete next step at all — it neither answers the question, gives a troubleshooting step, NOR asks for account details (DM requests DO count as a valid concrete next step, do not flag those)
- the customer message indicates this is a repeated/unresolved issue after already trying standard steps (e.g. "I already tried that", "still not working after doing X", "this is the third time")

Auto-handle in all other cases, including: how-to questions, informational answers, standard troubleshooting steps, requests to DM for account-specific investigation, and noise/thanks/praise needing no action.

Respond in this exact format:
DECISION: <AUTO_HANDLE or ESCALATE>
REASON: <one sentence>"""


def classify_intent(text, prior_context=None):
    context_block = ""
    if prior_context:
        turns = "\n".join(
            f'{"Customer" if c["inbound"] else "XboxSupport"}: "{c["text"]}"' for c in prior_context
        )
        context_block = f"Prior conversation turns (for context):\n{turns}\n\n"

    result = chat(
        [{"role": "user", "content": CLASSIFY_PROMPT.format(text=text, context_block=context_block)}],
        max_tokens=20,
    )
    result = result.strip().lower()
    for intent in INTENTS:
        if intent in result:
            return intent
    return "noise_other"


def draft_reply(text, intent, retriever, k=3):
    examples = retriever.query(text, k=k)
    examples = [e for e in examples if e[0].get("customer_text") != text][:k]

    grounding = "\n\n".join(
        f'Similar past case:\nCustomer: "{e[0]["customer_text"]}"\nXboxSupport replied: "{e[0]["brand_reply_text"]}"'
        for e in examples
    )

    prompt = f"""You are drafting a reply as XboxSupport's Twitter support account. Match the brand's real tone and style shown in the examples below.

{grounding}

New customer message (intent: {intent}): "{text}"

Draft a reply in XboxSupport's voice (concise, helpful, may end with an initials tag like real agents do). Output ONLY the reply text."""

    reply = chat([{"role": "user", "content": prompt}], max_tokens=150)
    return reply.strip(), examples


def decide_escalation(text, intent, reply):
    result = chat(
        [{"role": "user", "content": ESCALATE_PROMPT.format(intent=intent, text=text, reply=reply)}],
        max_tokens=100,
    )
    decision = "ESCALATE" if "ESCALATE" in result and "AUTO_HANDLE" not in result.split("REASON")[0] else "AUTO_HANDLE"
    reason = result.split("REASON:")[-1].strip() if "REASON:" in result else result.strip()
    return decision, reason


def run_pipeline(text, retriever, prior_context=None):
    intent = classify_intent(text, prior_context)
    reply, examples = draft_reply(text, intent, retriever)
    decision, reason = decide_escalation(text, intent, reply)
    return {
        "customer_text": text,
        "intent": intent,
        "drafted_reply": reply,
        "escalation_decision": decision,
        "escalation_reason": reason,
        "grounding_examples": [e[0]["customer_text"] for e in examples],
    }


if __name__ == "__main__":
    retriever = Retriever()
    text = sys.argv[1] if len(sys.argv) > 1 else "@XboxSupport my achievements aren't unlocking after I completed them"
    result = run_pipeline(text, retriever)
    print(json.dumps(result, indent=2, ensure_ascii=False))
