"""Decision policy: hard gates → model P(success) per candidate amount → expected value.

For each candidate charge (100%, 75%, 50%, 25% of overdue) we compute

    EV = P(success) × amount − (1 − P) × (fail_fee + harm_weight × amount)

and pick the best. If no candidate has positive EV (or P is below the floor) we skip.
The model scores; human-set policy limits always sit above it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import Settings
from app.model import TrainedModel
from app.world import FEATURES

CANDIDATE_FRACTIONS = (1.0, 0.75, 0.5, 0.25)


@dataclass
class Choice:
    action: str  # charge | partial | skip
    fraction: float
    amount: float
    probability: float
    expected_value: float


def feature_row(base: dict[str, float], charge_fraction: float) -> np.ndarray:
    values = {**base, "charge_fraction": charge_fraction}
    return np.array([float(values[f]) for f in FEATURES])


def expected_value(
    p: np.ndarray, amount: np.ndarray, settings: Settings, harm_weight: np.ndarray | float | None = None
) -> np.ndarray:
    harm = settings.harm_weight if harm_weight is None else harm_weight
    loss = settings.fail_fee + harm * amount
    return p * amount - (1 - p) * loss


def budget_limits(declines_left: int | None, settings: Settings) -> tuple[float, float]:
    """(min success probability, harm weight) for a customer's remaining decline budget.

    Schemes allow only a few declines a month (Mada 3, Visa/MC 6), so the last slots are
    worth more: the fewer left, the higher the bar to spend one. None = budget unknown.
    """
    if declines_left is None or declines_left >= 3:
        return settings.min_success_prob, settings.harm_weight
    if declines_left == 2:
        return settings.budget_min_prob_2_left, settings.harm_weight * settings.budget_harm_multiplier_2_left
    if declines_left == 1:
        return settings.budget_min_prob_1_left, settings.harm_weight * settings.budget_harm_multiplier_1_left
    return 1.01, settings.harm_weight  # 0 left: never attempt (the hard gate also catches this)


def schedule_min_prob(declines_left: int | None, settings: Settings) -> float:
    """Bar for a scheduled attempt: block hopeless ones, and guard the last decline."""
    if declines_left is not None and declines_left <= 1:
        return settings.schedule_min_prob_last_decline
    return settings.schedule_min_prob


def full_charge_bar(declines_left: int | None, settings: Settings, scheduled: bool) -> float:
    """Minimum P(success) to spend a decline on charging the full overdue."""
    if declines_left is not None and declines_left <= 1:
        if scheduled:
            return settings.schedule_min_prob_last_decline
        return settings.repayment_min_prob_last_decline
    return settings.schedule_min_prob if scheduled else settings.repayment_min_prob


def full_charge_gate(
    model: TrainedModel,
    bases: list[dict[str, float]],
    declines_left: list[int | None],
    settings: Settings,
    scheduled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Charge-in-full gate used at repayment moments and on scheduled days.

    The 30-day backtest showed partials don't pay off: the remainder falls back to cron,
    which charges on a worse day and fails. So the decision is binary — charge the full
    overdue now, or keep the decline for a better moment. Returns (P, bar, go).
    """
    rows = np.array([feature_row(b, 1.0) for b in bases])
    p = model.predict(rows)
    bar = np.array([full_charge_bar(d, settings, scheduled) for d in declines_left])
    return p, bar, p >= bar


def choose_many(
    model: TrainedModel,
    bases: list[dict[str, float]],
    overdue: np.ndarray,
    settings: Settings,
    declines_left: list[int | None] | None = None,
) -> tuple[list[Choice], np.ndarray]:
    """Vectorized policy over many customers. Returns choices + P matrix [n, candidates]."""
    n, k = len(bases), len(CANDIDATE_FRACTIONS)
    rows = np.array([feature_row(b, f) for b in bases for f in CANDIDATE_FRACTIONS])
    probs = model.predict(rows).reshape(n, k)
    fractions = np.array(CANDIDATE_FRACTIONS)
    amounts = np.minimum(np.round(overdue[:, None] * fractions[None, :], 2), settings.max_charge)
    limits = [budget_limits(d, settings) for d in (declines_left or [None] * n)]
    min_prob = np.array([m for m, _ in limits])
    harm = np.array([h for _, h in limits])
    ev = expected_value(probs, amounts, settings, harm[:, None])
    # Tiny partials are not worth the customer noise.
    too_small = (amounts < settings.min_charge) & (fractions[None, :] < 1.0)
    ev = np.where(too_small, -np.inf, ev)

    choices = []
    for i in range(n):
        j = int(np.argmax(ev[i]))
        p, amt, best = float(probs[i, j]), float(amounts[i, j]), float(ev[i, j])
        if best <= 0 or p < min_prob[i]:
            choices.append(Choice("skip", 0.0, 0.0, float(probs[i, 0]), best))
        else:
            action = "charge" if fractions[j] == 1.0 else "partial"
            choices.append(Choice(action, float(fractions[j]), amt, p, best))
    return choices, probs


def hard_gate(
    overdue_amount: float, fails_7d: int, settings: Settings, declines_left: int | None = None
) -> str | None:
    if overdue_amount <= 0:
        return "No eligible overdue balance"
    if declines_left is not None and declines_left <= 0:
        return "Decline budget used up for this month (scheme limit)"
    if fails_7d >= settings.max_fails_7d:
        return f"Attempt cap: {fails_7d} failed attempts in 7 days (cap {settings.max_fails_7d})"
    return None


def next_best_action(
    base: dict[str, float], probs_row: np.ndarray, declines_left: int | None = None
) -> str:
    if declines_left is not None and declines_left <= 0:
        return "No declines left this month — wait for next month's budget; send a soft reminder"
    if base["pm_health"] < 0.4:
        return "Ask customer to update payment method (in-app nudge); don't retry this card"
    if declines_left is not None and declines_left <= 2:
        return f"Save the last {declines_left} decline(s) for a stronger moment: next repayment or payday window"
    if not base["payday_window"]:
        return "Defer: retry in the payday window (27th–3rd) when funds are likelier"
    if probs_row.max() < 0.3:
        return "Send soft reminder; wait for next repayment event"
    return "Retry on next repayment event"


def describe(choice: Choice, probs_row: np.ndarray, overdue: float, currency: str,
             drivers: list[dict[str, Any]]) -> str:
    why = ", ".join(f"{d['direction']}{d['label']}" for d in drivers)
    full_p = probs_row[0]
    if choice.action == "charge":
        head = f"P(success) {full_p:.0%} — full charge {currency} {choice.amount:.2f}"
    elif choice.action == "partial":
        head = (
            f"Full {currency} {overdue:.2f} only {full_p:.0%} likely; "
            f"{choice.fraction:.0%} ({currency} {choice.amount:.2f}) is {choice.probability:.0%} "
            f"— best expected recovery, partial charge"
        )
    else:
        head = f"P(success) {full_p:.0%} at full amount; no charge clears the bar — skip"
    return f"{head}. Drivers: {why}"


def summarize_candidates(probs_row: np.ndarray, overdue: float, settings: Settings) -> list[dict]:
    fractions = np.array(CANDIDATE_FRACTIONS)
    amounts = np.minimum(np.round(overdue * fractions, 2), settings.max_charge)
    ev = expected_value(probs_row, amounts, settings)
    return [
        {
            "fraction": float(f),
            "amount": float(a),
            "probability": round(float(p), 4),
            "expected_value": round(float(e), 2),
        }
        for f, a, p, e in zip(fractions, amounts, probs_row, ev)
    ]
