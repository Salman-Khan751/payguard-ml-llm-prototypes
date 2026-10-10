# PayGuard Take-Home: Fraud Detection + LLM Complaint Triage (FastAPI)

## Layout
```
part_a_fraud/      notebook + models.py (tested helper code)
part_b_complaint/  app/ (FastAPI: main, schemas, llm, triage), tools.py, evaluation.ipynb, data/
part_c/memo.md
tests/             test_fraud.py, test_triage.py   (no network / API key needed)
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate     # .venv/ is git-ignored
pip install -r requirements.txt
cp .env.example .env                                  # add OPENAI_API_KEY
```

## Part A – Fraud Detection

This project uses machine learning to detect fraudulent credit card transactions. I compared Logistic Regression as a baseline with LightGBM, using class weighting to handle imbalanced data.

**Approach:**

* Cleaned the dataset and handled missing values, duplicates, and infinite values.
* Applied `log1p` transformation to transaction amounts.
* Split the data chronologically into 70% training, 15% validation, and 15% testing.
* Evaluated models using PR-AUC, ROC-AUC, precision, recall, and estimated business cost.
* Selected the decision threshold using validation data and an illustrative false-positive cost.

**Results:** LightGBM achieved a higher test PR-AUC than Logistic Regression in the reported results.

**Limitations:** The false-positive cost is an assumption, and model performance may change over time. Further validation and monitoring are needed before production use.

**How to run:**

1. Download `creditcard.csv` using `part_a_fraud/data/README.md`.
2. Place it in `part_a_fraud/data/`.
3. Run `part_a_fraud/fraud_detection.ipynb` from the `part_a_fraud/` directory.



## Part B - Complaint triage API
```bash
python -m part_b_complaint.tools fetch          # CFPB Consumer Complaint Database API: 300 complaints with narratives, 2024-01-01..2024-03-31
python -m part_b_complaint.tools labelsheet     # hand-label data/labeled_sample.csv BEFORE viewing model output
uvicorn part_b_complaint.app.main:app --reload  # docs at http://127.0.0.1:8000/docs
```
```bash
curl -X POST localhost:8000/triage -H 'content-type: application/json' \
  -d '{"narrative": "Someone used my card for a $900 purchase I never made."}'
```
Endpoints: `POST /triage`, `POST /triage/batch` (max 50), `GET /health`, `GET /metrics` (tokens + projected cost at 50k/month).
Then run `part_b_complaint/evaluation.ipynb` for accuracy, pipeline health, injection tests, and cost.

Design: strict JSON-schema output validated by Pydantic; complaint text treated as untrusted data (delimiter-stripped, truncated to 1,500 chars); a deterministic regex can only *escalate* to a human, never downgrade (it scans the full text, not the truncated text); any LLM/API/schema failure returns a safe fallback routed to a human instead of an error or silent drop.

### Part B approach and results

**Slice:** 300 CFPB complaints with narratives, 2024-01-01 to 2024-03-31, fetched by `python -m part_b_complaint.tools fetch` (stratified by product, so the same slice can be rebuilt).

**Taxonomy (7 routes):** fraud/unauthorized activity, billing/fees/disputes, account access/closure, loan/mortgage/debt collection, credit reporting/identity, service/other, legal/regulatory threat. Tie-break rules: a legal threat wins; scams go to fraud; loan-related fees go to loans.

**How I evaluate:** I labelled 50 complaints (category, urgency, requires_human) and compare the model with them. The number I care about most is `requires_human` recall, because a missed sensitive case (fraud claim, legal threat, hardship) is the costly mistake. I also track precision, the share sent to a human, LLM failure rate, latency, and a 6-case prompt-injection test.

**Results (run 2, after prompt and regex changes, n = 50):**

| Measure | Run 1 | Run 2 |
|---|---|---|
| Category accuracy | 0.74 | 0.84 |
| Urgency exact match | 0.68 | 0.72 |
| `requires_human` recall | 0.85 (missed 2) | 1.0 (missed 0) |
| `requires_human` precision | 0.31 | 0.50 |
| Injection tests passed | 5 / 6 | 6 / 6 |

Pipeline health on all 300: 0% LLM failures, median latency about 1.4 s, the safety regex overrode the model on 5%, and 67% of complaints were sent to a human.

**Read these numbers carefully:**
- I tuned the prompt and regex on the same 50 labels, so run 2 is optimistic.
- Claude drafted a first pass of the labels and I reviewed them (see `AI_USAGE.md`). The count of sensitive cases changed from 13 to 17 between runs, so run 1 and run 2 are not a perfectly fair comparison.
- The model still sends about twice as many complaints to a human as my labels say it should (about 68% vs 34% on the 50). That would not reduce a support team's workload yet.
- `legal_regulatory_threat` has only 1 labelled example and `service_quality_other` recall is 0.43, so rare-category scores are unreliable.

**Failure modes covered:** invented categories (enum validation), malformed JSON (schema check, retry, then fall back to a human), prompt injection (delimiter stripping, truncation, regex on the full text), and sensitive cases (regex can only escalate).

**Cost:** about $0.000156 per complaint, about $7.82 per month at 50,000 complaints with gpt-4o-mini (prices are unverified). This does not apply to an in-country self-hosted model, which has GPU cost instead.

**What I would measure at scale:** weekly human audit of a random sample of auto-routed complaints, missed-sensitive rate, escalation rate, category drift, override rate, and a fixed golden set re-run on every prompt or model change.

Run outputs are saved in `part_b_complaint/outputs/` (git-ignored). Re-running the notebook needs an `OPENAI_API_KEY`.

## Tests
```bash
python -m pytest -q
```

## What I cut and why
- **Part A:** no hyperparameter search, no ensembling, no neural network. The brief says not to, and a baseline vs one boosted model is enough to show the reasoning. No real-time feature demo; I describe it in the notebook instead. No confidence intervals or probability calibration (only 52 test frauds), which I would add next.
- **Part B:** no fine-tuning and no UI (out of scope). Only 50 hand labels, because careful labelling takes longer than coding. No second, untouched test set, so run 2 is optimistic. I did not use the CFPB's own product tags as a cheap extra label source. I did not try a local/open model.
- **Part C:** one memo for Part B only, with a text diagram instead of a drawn one.

## Part C
See `part_c/memo.md`.
