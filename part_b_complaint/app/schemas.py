from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Category(str, Enum):
    fraud_unauthorized_activity = "fraud_unauthorized_activity"
    billing_fees_disputes = "billing_fees_disputes"
    account_access_or_closure = "account_access_or_closure"
    loan_mortgage_debt_collection = "loan_mortgage_debt_collection"
    credit_reporting_identity = "credit_reporting_identity"
    service_quality_other = "service_quality_other"
    legal_regulatory_threat = "legal_regulatory_threat"


CATEGORIES = [c.value for c in Category]
Urgency = Literal["low", "medium", "high"]


class LLMTriage(BaseModel):
    """Exactly what the model must return (validated before anything trusts it)."""
    category: Category
    urgency: Urgency
    summary: str = Field(max_length=500)
    requires_human: bool
    confidence_note: str = Field(max_length=300)


class TriageRequest(BaseModel):
    narrative: str = Field(min_length=1, max_length=20_000)


class BatchRequest(BaseModel):
    narratives: list[TriageRequest] = Field(min_length=1, max_length=50)


class TriageResponse(LLMTriage):
    llm_failed: bool = False        # model/API failed -> safe fallback, routed to human
    forced_by_rule: bool = False    # regex safety net flipped requires_human to True
    latency_s: float = 0.0


class BatchResponse(BaseModel):
    results: list[TriageResponse]


class Metrics(BaseModel):
    calls: int
    tokens_in: int
    tokens_out: int
    est_cost_usd: float
    est_cost_per_complaint_usd: float
    est_cost_50k_per_month_usd: float
