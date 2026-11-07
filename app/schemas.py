from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DecisionAction(str, Enum):
    CHARGE = "charge"
    PARTIAL = "partial"
    SKIP = "skip"


class FeatureSet(BaseModel):
    """Matches rpm-autopay-service FeatureSet; extra keys are ignored."""

    model_config = ConfigDict(extra="ignore")

    days_overdue: int
    pm_health: float = Field(ge=0, le=1)
    fails_7d: int = Field(ge=0)
    recent_success: float = Field(ge=0, le=1)
    repayment_strength: float = Field(default=1.0, ge=0, le=1)
    overdue_amount: Optional[float] = None
    amount_paid: Optional[float] = None
    # Scheme declines left this month on the card (Mada 3, Visa/MC 6). Omit if unknown.
    declines_left: Optional[int] = Field(default=None, ge=0)
    # Card on file. Omitted = baseline (Mada, plastic, debit).
    card_network: Optional[Literal["mada", "visa", "mastercard"]] = None
    card_token: Optional[Literal["plastic", "apple_pay"]] = None
    card_funding: Optional[Literal["debit", "credit", "prepaid"]] = None


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_id: str
    customer_id: str
    event_type: str = "repayment.succeeded"
    amount_paid: float
    overdue_amount: float
    currency: str = "SAR"
    features: FeatureSet
    as_of: Optional[date] = None  # defaults to today; drives the payday-window feature


class Candidate(BaseModel):
    fraction: float
    amount: float
    probability: float
    expected_value: float


class Driver(BaseModel):
    feature: str
    label: str
    value: float
    impact: float
    direction: str


class DecisionResponse(BaseModel):
    # Contract fields consumed by rpm-autopay-service
    action: DecisionAction
    amount: float
    score: float
    reason: str
    baseline_action: DecisionAction = DecisionAction.CHARGE
    # Extras for UI / ops
    expected_value: Optional[float] = None
    gate: Optional[str] = None
    next_best_action: Optional[str] = None
    drivers: list[Driver] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    model_version: str = ""
    idempotent_replay: bool = False
    declines_left: Optional[int] = None
    min_success_prob_applied: Optional[float] = None
    card: Optional[str] = None


class BacktestRequest(BaseModel):
    n: int = Field(default=1000, ge=50, le=20000)
    seed: int = 2026
    harm_weight: Optional[float] = Field(default=None, ge=0, le=5)  # policy dial override


class StrategyResult(BaseModel):
    strategy: str
    attempts: int
    successes: int
    failed_attempts: int
    success_rate: float
    recovered: float
    fail_fees: float
    net_recovered: float
    recovered_per_attempt: float


class BacktestResponse(BaseModel):
    customers: int
    total_overdue: float
    strategies: list[StrategyResult]
    summary: str
    note: str


class MonthBacktestRequest(BaseModel):
    n: int = Field(default=2000, ge=50, le=10000)
    seed: int = 2026
    decline_budget: int = Field(default=3, ge=1, le=10)  # Mada
    decline_budget_visa_mc: int = Field(default=6, ge=1, le=12)  # Visa / Mastercard
    harm_weight: Optional[float] = Field(default=None, ge=0, le=5)


class MonthStrategyResult(BaseModel):
    strategy: str
    attempts: int
    failed_attempts: int
    success_rate: float
    recovered: float
    recovery_rate: float
    fail_fees: float
    customers_fully_cleared: float
    customers_out_of_declines: float
    avg_day_first_collection: Optional[float] = None


class MonthBacktestResponse(BaseModel):
    customers: int
    days: int
    decline_budget: int
    decline_budget_visa_mc: int
    total_overdue: float
    strategies: list[MonthStrategyResult]
    summary: str
    note: str


class ModelInfo(BaseModel):
    version: str
    features: list[str]
    coefficients: dict[str, float]
    metrics: dict[str, Any]
    policy: dict[str, Any]
