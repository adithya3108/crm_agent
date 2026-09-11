# XboxSupport AI Support Agent: Report

## 1. Problem framing

**Brand chosen: XboxSupport.** Selected over SpotifyCares and AmericanAir after sampling threads from all three (see decision log #1). XboxSupport gives substantive in-thread technical answers rather than deflecting everything to DM, which matters because the assignment asks for replies "grounded in how the brand has historically resolved similar issues." SpotifyCares and AmericanAir mostly hand off to DM for anything account-specific, meaning the real resolution never appears in the public dataset.

**What "good" means for this agent:**
- **Intent classification** is good if it beats simple baselines by a wide margin and its errors cluster in genuinely ambiguous cases (short follow-ups, borderline categories), not systematic blind spots.
- **A good reply** matches the brand's actual resolution pattern (which, for XboxSupport, is frequently "ask for Gamertag + details via DM," not a failure, a norm) and gives a concrete next step, whether a troubleshooting step, an answer, or a legitimate account-investigation handoff.
- **A good escalation decision** is not "auto-handle whenever possible." The assignment explicitly wants a stated reason and a system we can trust. We treat recall on true escalations as more important than precision (missing an escalation is worse than an unnecessary human review), but not at the cost of escalating nearly everything.

**What we chose not to build:**
- No fine-tuned model. Used prompted Claude (via OpenRouter) for all three pipeline stages, since 23k threads is enough for good retrieval grounding but not enough to justify a fine-tune, and it keeps the repo reproducible without a training step.
- No dense/embedding retrieval. Used TF-IDF for reply grounding (decision log #6), trading some retrieval quality for zero extra API cost and simplicity within the 15-minute reproduction budget.
- No multi-turn dialogue state beyond the immediate prior context (up to 3 turns). The dataset is mostly single-exchange threads, and deeper context tracking wasn't worth the complexity for the volume of multi-turn threads actually present.

## 2. Results vs. baselines

Two baselines, both implemented and run on the same 200-example golden set as the pipeline:
- **Trivial**: always predicts the majority-class intent, always outputs AUTO_HANDLE.
- **Simple**: keyword-rule intent classifier + keyword-rule escalation (see `src/eval_harness.py`).

The pipeline itself went through 3 iterations as failure analysis surfaced problems, reported here as the honest before/after story rather than only the final number, because the iteration itself demonstrates the precision/recall tradeoffs that matter for this kind of system.

| Metric | Trivial | Simple (keyword) | Pipeline v1 | Pipeline v2 | Pipeline v3 |
|---|---|---|---|---|---|
| Intent accuracy | 24.5% | 36.5% | 77.0% | 78.0% | 78.5% |
| Escalation accuracy | 81.5% | 79.5% | 62.0% | 84.5% | 78.0% |
| Escalation precision | 0% | 41.7% | 32.1% | 61.5% | 42.9% |
| Escalation recall | 0% | 27.0% | 94.6% | 43.2% | 56.8% |
| Escalation F1 | 0 | 32.8% | 47.9% | 50.8% | 48.8% |
| Reply groundedness (LLM judge, 1-5) | n/a | n/a | 3.44 | 3.48 | 3.48 |
| Reply helpfulness (LLM judge, 1-5) | n/a | n/a | 3.47 | 3.50 | 3.50 |
| Reply tone (LLM judge, 1-5) | n/a | n/a | 4.39 | 4.41 | 4.42 |

**What changed between versions, and why (full detail in decision log #8):**

*v1 to v2*: v1's escalation prompt treated "reply asks for a DM/more info" as a sign of an unresolved/vague answer. But DM-handoff is XboxSupport's *normal, successful* resolution pattern, so v1 was escalating almost everything (94.6% recall) at the cost of precision (32.1%). v2 explicitly told the model not to treat DM-requests as escalation-worthy, restricting escalation to 2 intent categories (enforcement_ban, billing_subscription) plus explicit anger/threats. This fixed precision (61.5%) but recall collapsed (43.2%): the rule became too narrow, missing legitimate escalations outside those 2 categories.

*v2 to v3*: hypothesized the recall gap was cases where the drafted reply gives no concrete next step, or where the customer signals a repeated/already-tried-that issue. Added both as explicit escalate triggers, while keeping v2's rule that a DM request still counts as a valid next step. Result: recall improved (43.2% to 56.8%) as intended, but precision dropped further than recall gained (61.5% to 42.9%), so F1 ended up essentially flat, slightly below v2 (50.8% to 48.8%). Note: an earlier run of this same v3 prompt appeared to produce identical numbers to v2 due to a network failure that silently left stale results in place; the numbers above are from a verified, fully-completed re-run, confirmed by diffing raw per-example decisions against v2 (23 of 200 decisions actually changed). This is disclosed here because it is exactly the kind of silent-failure risk that matters for trusting an eval pipeline.

Net effect across all three versions: F1 moved from 47.9% (v1) to 50.8% (v2) to 48.8% (v3), a small, noisy improvement over the original version with no version clearly dominating on both precision and recall simultaneously. This is a genuine finding, not a clean success story: single free-text escalation prompts appear to trade precision for recall in ways that are hard to control without a more structured decision mechanism (see failure mode 5 and next-steps #4).

**Human-vs-judge agreement on reply quality.** The assignment asks for evidence of how well the LLM-as-judge agrees with a human. The author blind-scored 30 drafted replies (grounded, helpful, tone, 1-5 each) without seeing the judge's scores, then compared:

| Dimension | Exact match | Within 1 point | Correlation | Mean human score | Mean judge score |
|---|---|---|---|---|---|
| Grounded | 26.7% | 73.3% | 0.35 | 4.60 | 3.67 |
| Helpful | 26.7% | 73.3% | 0.36 | 4.23 | 3.30 |
| Tone | 43.3% | 100.0% | ~0.0 | 4.93 | 4.43 |

Two findings here matter more than the headline percentages. First, there is a **systematic bias**: the human scorer rated every dimension roughly 0.5-0.9 points higher than the judge on average, meaning the judge is a consistently harsher critic than a human reader in this case. Second, the **correlation is weak** even setting bias aside (0.35, 0.36, ~0.0), meaning the judge does not reliably reproduce a human's *relative* ranking of which replies are better or worse than others, not just their absolute scores. The near-zero tone correlation is partly an artifact of human scores clustering tightly at 4-5 (little variance to correlate against), which is itself informative: a human reading these replies rarely perceives tone as a problem, while the judge assigns more spread. Practical implication: the judge's 1-5 groundedness/helpfulness numbers reported throughout this document (around 3.4-3.5) should be read as a relative signal for comparing pipeline versions against each other, not as a calibrated absolute measure of reply quality as a human would perceive it; a human reviewer would likely rate the same replies meaningfully higher.

## 3. Failure analysis: top failure modes with real examples

**1. Escalation logic is sensitive to how "concrete next step" is defined.** The single biggest lever in this whole system turned out to be one sentence in the escalation prompt. Real example (v1, false positive): customer says "my achievements aren't unlocking," drafted reply says "Can you DM us your gamertag so we can look into this?" v1 escalated this because the reply "didn't resolve the issue in-line." But this is XboxSupport's standard successful pattern. *Hypothesis: any escalation rule built by inference over reply text needs an explicit list of what a "resolved" pattern looks like for this specific brand, not a generic vagueness heuristic.*

**2. Short follow-up messages need conversational context to classify.** Example: "Still doing it" was classified as `noise_other` without context, but is clearly `game_app_crash` when the prior turn ("my xbox freezes when I try to launch Fifa") is included. Fixed in v2/v3 by feeding `prior_context`; only partial fix since only ~52% of golden examples have available prior context in the dataset (many customer messages are the first, unthreaded turn).

**3. Retrieval-grounding gaps lead to confident hallucination.** Example: asked about Final Fantasy XIV availability on Xbox, the model answered confidently and specifically despite TF-IDF retrieval surfacing no closely relevant historical example, judged 1/5 on both groundedness and helpfulness. *Hypothesis: TF-IDF similarity score should gate generation. Below a threshold, the reply should default to an honest "let me check on that" plus escalate, rather than free-generating from the model's own (unverified, possibly stale) knowledge.* Not yet implemented; see "next week" section.

**4. Intent confusion cluster: game_app_crash / hardware_issue / howto_general.** These three intents share vocabulary (both hardware and software problems get described as "not working," "freezing," etc. by customers who don't distinguish the cause themselves): 4+ misclassifications each direction between these three intents. *Hypothesis: needs few-shot examples per intent, or a "root cause unclear" secondary intent for cases where even a human couldn't tell from the tweet alone.*

**5. The v3 fix traded false negatives for false positives instead of fixing the underlying gap.** v3 (200 examples: 16 false negatives, 28 false positives) added "no concrete next step" and "repeated/unresolved issue" as escalate triggers to catch cases like these still-missed false negatives, with the author's own labeling notes:
   - *"I pre ordered COD WWII... didn't get my preorder bonus"*, note: "missing preorder bonus content." Still auto-handled: the drafted reply sounds complete ("your bonuses should be included automatically, can you check...") so it doesn't trip the "no concrete next step" trigger, even though the underlying issue (paid content not delivered) is escalate-worthy.
   - *"every time I try to change my home Xbox... entered the code but that wouldn't work either"*, note: "verification code failing, needs account investigation." The reply already recommends chat support, which reads as a resolved handoff to the trigger, so it's still not flagged.

   But the same triggers also newly over-fired on cases that should stay auto-handled, for example: *"When is cortana going to be available in Australia... no mention of if/when"* (a `howto_general` question with a complete, honest "we don't have a timeline" answer) got escalated, because a complete-but-unsatisfying answer looks similar to "no concrete next step" from the model's perspective.

   *Root cause: "does this reply look complete" is a surface-level signal that doesn't distinguish between "this genuinely resolves the case" and "this sounds complete but leaves the customer's actual problem (lost content, failed verification) unaddressed." Fixing this needs either (a) a severity/repeat-attempt classifier as a signal parallel to and independent of intent and reply text, not inferred from the drafted reply's phrasing, or (b) a rubric-based multi-factor check (repeat-attempt history, financial/content loss, account-specific investigation needed) scored independently rather than folded into one holistic LLM judgment of the reply.*

## 4. What is misleading about my headline numbers

- **Escalation accuracy alone is a trap.** On this imbalanced golden set (163 auto / 37 escalate, roughly an 82/18 split), the trivial "always auto-handle" baseline scores 81.5% accuracy, higher than pipeline v1's 62%, while having zero recall and being useless. Accuracy is the wrong headline metric here; F1 (which still shows the pipeline beating baselines) is the fairer one, and even that number moved less than accuracy did across v1 to v2, which is the more honest signal.
- **Golden-set labeling has a circularity problem.** All 200 golden labels were LLM-drafted (same model family as the pipeline) and then reviewed by the author. Correlated blind spots between the labeler and the classifier likely inflate intent accuracy relative to what fully independent human labels would show. This is a known, disclosed limitation, not swept under the rug; see decision log #7.
- **The v1 to v2 to v3 "improvement" story is really a tradeoff story, not a clean win.** F1 moved 47.9 to 50.8 to 48.8 across the three versions: essentially flat overall, while precision and recall individually swung by 20-30 points each iteration. No version dominates on both metrics simultaneously. Anyone citing only the final F1 number without this context would be misled into thinking the system converged to something stable, when really each iteration just picked a different point on the same precision/recall tradeoff curve.
- **A background eval run silently failed once during this project and produced stale-but-plausible-looking numbers** (an old result file was read as if it were a completed new run, because a network failure happened before the script could overwrite it). This was caught only by explicitly diffing per-example predictions between runs, not by looking at the summary numbers, which looked entirely reasonable on their own. This is disclosed here as a concrete illustration of why "the number looks fine" is not sufficient evidence that a pipeline ran correctly.
- **Reply tone consistently outscores groundedness/helpfulness** (4.4 vs roughly 3.5 out of 5). A support agent that "sounds right" but isn't reliably correct is a specific, identifiable risk for auto-handling; the tone score alone would look like a success story if reported without the other two dimensions.
- **The LLM-judge quality scores (around 3.4-3.5/5) understate how a human actually perceives these replies.** The human-vs-judge agreement check (see section 2) found the judge scores roughly 0.5-0.9 points lower than a human on the same replies, with only weak correlation (0.35 or less) to human ranking. Read in isolation, the ~3.5/5 groundedness/helpfulness numbers look mediocre; a human reading the same replies rated them closer to 4.2-4.6/5 on average. The judge numbers are still useful for comparing pipeline versions against each other, but should not be quoted as an absolute quality measure without this caveat.
- **The eval sample (200 examples, drawn from 23,235 threads for one brand)** is not necessarily representative of rare-but-high-stakes cases (e.g. genuine legal/safety escalations). These are underrepresented in a random sample, and the system has not been stress-tested against adversarial or edge-case inputs.

## 5. What I'd do with one more week

1. Replace TF-IDF retrieval with sentence embeddings plus a similarity-gated fallback (see failure mode 3) to reduce confident hallucination on under-grounded cases.
2. Add few-shot examples per intent to the classifier prompt, targeted at the game_app_crash/hardware_issue/howto_general confusion cluster (failure mode 4).
3. Run a 4th escalation-prompt iteration with the specific goal of raising recall back toward 70-80% without re-collapsing precision; likely needs a structured rubric (checklist of escalate triggers scored independently) rather than one free-text prompt.
4. Expand the golden set specifically with rare/high-stakes cases (harassment, safety, minors) that a random 200-sample likely underrepresents, since these matter disproportionately for a real auto-handle decision.
5.  next step is recalibrating the judge prompt to reduce the systematic negative bias found, e.g. by anchoring the rubric with concrete examples of what a 5/5 vs 3/5 reply looks like for this brand, rather than leaving the scale to the judge's own interpretation.

## 6. Reproduction

See `README.md` for exact steps. Headline results reproducible in under 15 minutes on a subsample of the dataset.
