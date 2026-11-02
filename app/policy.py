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


def expected_value(p: np.ndarray, amount: np.ndarray, settings: Settings) -> np.ndarray:
    loss = settings.fail_fee + settings.harm_weight * amount
    return p * amount - (1 - p) * loss


def choose_many(
    model: TrainedModel,
    bases: list[dict[str, float]],
    overdue: np.ndarray,
    settings: Settings,
) -> tuple[list[Choice], np.ndarray]:
    """Vectorized policy over many customers. Returns choices + P matrix [n, candidates]."""
    n, k = len(bases), len(CANDIDATE_FRACTIONS)
    rows = np.array([feature_row(b, f) for b in bases for f in CANDIDATE_FRACTIONS])
    probs = model.predict(rows).reshape(n, k)
    fractions = np.array(CANDIDATE_FRACTIONS)
    amounts = np.minimum(np.round(overdue[:, None] * fractions[None, :], 2), settings.max_charge)
    ev = expected_value(probs, amounts, settings)
    # Tiny partials are not worth the customer noise.
    too_small = (amounts < settings.min_charge) & (fractions[None, :] < 1.0)
    ev = np.where(too_small, -np.inf, ev)

    choices = []
    for i in range(n):
        j = int(np.argmax(ev[i]))
        p, amt, best = float(probs[i, j]), float(amounts[i, j]), float(ev[i, j])
        if best <= 0 or p < settings.min_success_prob:
            choices.append(Choice("skip", 0.0, 0.0, float(probs[i, 0]), best))
        else:
            action = "charge" if fractions[j] == 1.0 else "partial"
            choices.append(Choice(action, float(fractions[j]), amt, p, best))
    return choices, probs


def hard_gate(overdue_amount: float, fails_7d: int, settings: Settings) -> str | None:
    if overdue_amount <= 0:
        return "No eligible overdue balance"
    if fails_7d >= settings.max_fails_7d:
        return f"Attempt cap: {fails_7d} failed attempts in 7 days (cap {settings.max_fails_7d})"
    return None


def next_best_action(base: dict[str, float], probs_row: np.ndarray) -> str:
    if base["pm_health"] < 0.4:
        return "Ask customer to update payment method (in-app nudge); don't retry this card"
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
        head = f"P(success) {full_p:.0%} at full amount; no charge has positive expected value — skip"
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
