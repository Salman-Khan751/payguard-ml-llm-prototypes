import logging
import re
import time
from typing import Callable

from . import llm
from .schemas import Category, LLMTriage, TriageResponse

log = logging.getLogger("triage")

MAX_CHARS = 1500  # truncating long narratives is the biggest token saver

# Deterministic safety net: zero tokens, never lets a sensitive case auto-close.
SENSITIVE = re.compile(
    r"\b(lawyer|attorney|sue|suing|lawsuit|court|cfpb|regulator|identity theft|fraud|"
    r"scam|stolen|unauthori[sz]ed|police|harass\w*|threat\w*|"
    # hardship / vulnerable customers: the model alone missed these in evaluation
    r"grief|bereave\w*|passed away|died|lost my job|laid off|hospital\w*|cancer|"
    r"disabled|disability|terminal\w*|(?:i was|i am|been) sick)\b", re.I)

# Any variant of the delimiter tag (case, spaces, attributes) so text cannot break out.
_TAG = re.compile(r"<\s*/?\s*complaint[^>]*>", re.I)


def sanitise(text: str) -> str:
    return _TAG.sub("", text)[:MAX_CHARS]


def triage_one(text: str, llm_call: Callable[[str], LLMTriage] | None = None) -> TriageResponse:
    llm_call = llm_call or llm.call_llm   # resolved at call time so tests can patch it
    t0 = time.time()
    # Rule runs on the FULL text: truncation must not hide a late "I will sue you".
    sensitive = bool(SENSITIVE.search(text))
    failed = False
    try:
        out = llm_call(sanitise(text)).model_dump()
    except Exception as e:
        log.warning("LLM failure, routing to human: %s", e)
        failed = True
        out = {"category": Category.service_quality_other, "urgency": "medium", "summary": "",
               "requires_human": True,
               "confidence_note": "LLM failed or returned malformed output; routed to human."}
    forced = False
    if sensitive and not out["requires_human"]:
        out["requires_human"], forced = True, True   # rule overrides the model
    return TriageResponse(**out, llm_failed=failed, forced_by_rule=forced,
                          latency_s=round(time.time() - t0, 3))