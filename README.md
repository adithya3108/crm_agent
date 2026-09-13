# XboxSupport AI Support Agent

An AI customer-support agent for XboxSupport (Twitter), built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) Kaggle dataset. For each incoming customer message it: (1) classifies intent into one of 10 brand-specific categories, (2) drafts a reply grounded in XboxSupport's own historical resolutions via retrieval, and (3) decides whether to auto-handle or escalate to a human, with a stated reason.

See `REPORT.md` for the full writeup (problem framing, results vs. baselines, failure analysis, limitations) and `decision_log.md` for the non-obvious decisions made along the way.

## Setup (5 min)

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=anthropic/claude-sonnet-4.5
```

Get a key at [openrouter.ai](https://openrouter.ai). Any OpenRouter-hosted chat model works via `OPENROUTER_MODEL`.

**Note on model used:** the headline results in `REPORT.md` (78.5% intent accuracy, escalation F1 ~51%, etc.) were generated with `anthropic/claude-sonnet-4.5`. The most recent testing in this repo, however, was done with `meta-llama/llama-3.3-70b-instruct` (open-source, ~10-20x cheaper per call) on a 20-example subsample to sanity-check the pipeline still runs correctly on a different model; see `data/eval_report_llama_trial.json` for those numbers. Results differ somewhat between models (expected, given they're different models), so if you're trying to reproduce the exact numbers in `REPORT.md`, set `OPENROUTER_MODEL=anthropic/claude-sonnet-4.5`.

## Reproduce the headline results (under 15 min)

The full dataset (~3M tweets, 516MB) is not checked into this repo. All scripts below work on a small, already-included subsample so you don't need to re-download or re-process the full dataset to verify results.

**1. (Included) Reconstructed XboxSupport threads and golden eval set.**
`data/xbox_threads.jsonl` (23,235 threads) and `data/golden_labeled.csv` (200 hand-reviewed examples) are pre-built and included. Skip to step 2.

*If you want to rebuild from scratch instead (adds ~10 min + a Kaggle account):*
```bash
python scripts/download_data.py      # downloads full dataset via kagglehub, needs KAGGLE_API_TOKEN env var
python scripts/build_threads.py      # reconstructs XboxSupport threads -> data/xbox_threads.jsonl
python src/retrieval.py              # builds TF-IDF index -> data/retrieval_index.pkl
```

**2. Run the pipeline on a single example (~10 sec):**
```bash
python src/pipeline.py "@XboxSupport my achievements aren't unlocking after I completed them"
```
Outputs classified intent, a grounded drafted reply, and the escalation decision with reason as JSON.

**3. Run the eval harness on a subsample (~2-3 min for 30 examples):**
```bash
python src/eval_harness.py 30
```
This scores the pipeline plus two baselines (trivial majority-class, keyword-rule) against 30 of the 200 golden examples on intent accuracy, escalation precision/recall/F1, and LLM-judge reply-quality scores. Results are written to `data/eval_report.json` and full per-example output to `data/eval_raw_results.json`.

**4. Run on the full 200-example golden set (~15-20 min, optional):**
```bash
python src/eval_harness.py
```
This is what produced the numbers reported in `REPORT.md`.

## Project structure

```
src/
  llm.py              OpenRouter client with retry logic
  retrieval.py         TF-IDF retrieval index over historical (customer, reply) pairs
  pipeline.py           3-stage pipeline: classify_intent -> draft_reply -> decide_escalation
  eval_harness.py       Scores pipeline + 2 baselines against the golden set
scripts/
  download_data.py       Downloads the Kaggle dataset via kagglehub
  build_threads.py        Reconstructs XboxSupport (customer, reply) threads with prior context
  sample_intents.py       Sampled messages used to hand-derive the intent taxonomy
  sample_golden.py         Sampled 200 examples for the golden eval set
  draft_labels.py          LLM-drafts intent labels as a starting point for hand-labeling
data/
  xbox_threads.jsonl        23,235 reconstructed threads (included)
  golden_labeled.csv         200-example golden eval set: AI-drafted, human-reviewed (included)
  eval_report.json           Latest eval run's summary metrics (Claude Sonnet, matches REPORT.md)
  eval_raw_results.json      Latest eval run's full per-example output (drafted replies, judge scores)
  eval_report_llama_trial.json   20-example sanity-check run on meta-llama/llama-3.3-70b-instruct
REPORT.md              Full writeup
decision_log.md          10-15 non-obvious decisions and why
```

## Known limitations (see REPORT.md section 4 for the full "what's misleading" discussion)

- Golden set labels were LLM-drafted then human-reviewed, not written from scratch by a human; this likely inflates intent accuracy somewhat due to correlated model bias between the labeler and the classifier.
- Retrieval uses TF-IDF, not embeddings; misses some semantic near-duplicates.
- Escalation logic went through 3 documented iterations and still trades precision against recall rather than improving both; see the version history in `REPORT.md` section 2.
