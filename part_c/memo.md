# Memo: Taking the Complaint-Triage Prototype to Production

**To:** CTO, PayGuard
**From:** Consulting Data Scientist
**Scope:** Part B, LLM complaint triage

## 1. Where the prototype stands

I ran the prototype on 300 CFPB complaints (Jan–Mar 2024) and checked 50 of them against my own labels. The sample is small, and I tuned the prompt on those same 50, so treat run 2 as optimistic.

| Measure | Run 1 | Run 2 (after fixes) |
|---|---:|---:|
| Category accuracy | 74% | 84% |
| Urgency exact match | 68% | 72% |
| Sensitive cases caught (`requires_human` recall) | 85% (missed 2) | 100% (missed 0) |
| Human-review precision | 0.31 | 0.50 |
| Injection tests passed | 5 of 6 | 6 of 6 |

On all 300 complaints: no malformed outputs or LLM failures, median latency about 1.4 s, and the safety rules overrode the model on 5%. API cost is about $0.000156 per complaint, roughly **$7.82 per month at 50,000 complaints** (prices unverified).

**Main problem:** the model is still too cautious. It sends 67% of complaints to a human, and on my 50 labels it flags about 68% where I believe only 34% need it. That would not reduce the support team's workload. This must be fixed before any automation. Run 1 and run 2 are also not perfectly comparable, because my count of sensitive labels changed from 13 to 17 after I reviewed them.

## 2. Architecture

```
Complaint (app / email / web)
        |
  [Intake + PII masking]            in-country
        |
  [Sanitiser: strip tags, truncate]
        |                  \
  [LLM, in-country,         [Safety regex on FULL text]
   strict JSON, retry x2]    can only force "human"
        |                  /
  [Validator: schema + category check]
        |        failure -> human queue
  [Router]
     |-- sensitive --> priority human queue (never auto-closed)
     |-- other     --> team queue with summary + confidence note
        |
  [Log store: hashes, outputs, model/prompt version, human decision]
        |
  [Dashboards + alerts]
```

Rules already in the prototype: complaint text is data, not instructions; the regex can only escalate; any failure goes to a human instead of being dropped.

## 3. Monitoring: what breaks silently

| Silent failure | How I would catch it |
|---|---|
| Model or vendor changes behaviour | Fixed golden set re-run nightly and on every prompt/model change; alert on a drop |
| Escalation rate drifts (mine is already too high) | Daily escalation rate against an agreed band |
| New complaint types or scams | Weekly category mix; growth in "service/other" means the taxonomy is stale |
| Missed sensitive cases | Weekly human audit of 100 random auto-routed complaints; this is the top metric |
| Agents rubber-stamp the model | Track override rate; near zero for weeks is a warning |
| Prompt injection | Keep the injection tests in the nightly run; flag instruction-like text |
| Latency or failure creep | p95 latency and fallback rate |

## 4. Privacy, residency, retention

- **Residency:** the regulator requires data to stay in-country, so the US-hosted API in my prototype is not acceptable for real complaints. Production needs an open-weight model in an in-country data centre or cloud region, or a vendor that contractually guarantees in-country processing. Compliance must approve the choice. An open model may be less accurate, so I would re-run the evaluation on it.
- **Cost:** the $7.82 figure does not carry over. Self-hosting means GPU cost. Volume is low (about one complaint a minute on average), so I expect a modest server, but I would measure it.
- **PII:** mask names and account/card numbers before the model sees them where possible. Keep raw text out of monitoring logs.
- **Retention and access:** keep text only as long as the regulator requires, then delete. Role-based access, and log every human override.

## 5. Rollout

1. **Weeks 1–4, shadow mode:** the model runs, nobody sees it. Compare with what agents did and build the golden set.
2. **Weeks 5–8, assist mode:** agents see the suggested category and summary, and every complaint still gets human review. Measure time saved and override rate.
3. **Weeks 9–14, limited automation:** auto-route only low-risk categories with no sensitive keywords, and audit a 10% sample. Sensitive complaints stay human.
4. **Ongoing:** widen one category at a time, only while the audited error rate stays under the agreed limit.

**Gates (my proposal, to agree with PayGuard):** sensitive-case recall of at least 98% on the golden set, escalation rate near the true share of sensitive cases, and no legal threat or fraud allegation ever auto-closed.

## 6. What I need from PayGuard

- 5,000+ historical complaints with the routing and outcome agents actually used.
- Your real team list and routing rules, to replace my taxonomy.
- A definition of "sensitive" approved by Legal and Compliance, with the regulator's response deadlines.
- Written confirmation of what "in-country" means for hosting and vendors.
- 2–3 agents for about 2 hours a week to label and review.
- Target response times, and one owner for monitoring alerts.

## 7. Limits of my evidence

Only 50 labels, tuned against, with Claude's first-pass labels reviewed by me (see `AI_USAGE.md`). One legal-threat example. Six injection tests. I would not recommend automation until a larger labelled set exists and shadow mode confirms the results.
