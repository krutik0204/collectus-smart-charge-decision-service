"""Synthetic "world" — hidden ground truth for whether an autocharge succeeds.

Used for two things only:
1. Generating the historical charge-attempt dataset the model trains on.
2. Simulating outcomes in backtests (always-charge vs smart-charge).

The model never sees these coefficients; it has to learn them from history.
Replace this module with real Tamara autocharge history for production.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

FEATURES = [
    "pm_health",
    "recent_success",
    "fails_7d",
    "days_overdue",
    "repayment_strength",
    "charge_fraction",
    "trigger_event",
    "payday_window",
]

DECLINE_CODES = ["insufficient_funds", "do_not_honor", "card_expired", "limit_exceeded"]


def payday_window(day_of_month: int) -> int:
    """KSA government salaries land around the 27th; funds are highest 27th → 3rd."""
    return 1 if day_of_month >= 27 or day_of_month <= 3 else 0


def true_success_probability(x: dict[str, np.ndarray]) -> np.ndarray:
    z = (
        -2.0
        + 3.0 * x["pm_health"]
        + 1.8 * x["recent_success"]
        - 0.45 * x["fails_7d"]
        - 0.015 * x["days_overdue"]
        + 1.2 * x["repayment_strength"]
        - 2.0 * x["charge_fraction"]
        + 0.6 * x["trigger_event"]
        + 0.5 * x["payday_window"]
    )
    return 1.0 / (1.0 + np.exp(-z))


def sample_customers(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Customer state at the moment an overdue balance exists."""
    pm_health = rng.beta(4, 2, n)
    recent_success = np.clip(
        0.6 * pm_health + 0.4 * rng.beta(2, 2, n) + rng.normal(0, 0.05, n), 0, 1
    )
    fails_7d = np.minimum(rng.poisson(0.2 + 3.0 * (1 - pm_health) ** 2), 8)
    days_overdue = rng.integers(1, 91, n)
    overdue_amount = np.round(np.clip(rng.lognormal(np.log(250), 0.6, n), 20, 5000), 2)
    return {
        "pm_health": np.round(pm_health, 3),
        "recent_success": np.round(recent_success, 3),
        "fails_7d": fails_7d,
        "days_overdue": days_overdue,
        "overdue_amount": overdue_amount,
    }


def generate_history(n: int, seed: int = 42, today: date | None = None) -> list[dict]:
    """Historical autocharge attempts: ~35% event-triggered, the rest cron."""
    rng = np.random.default_rng(seed)
    today = today or date.today()
    x = sample_customers(n, rng)

    trigger_event = (rng.random(n) < 0.35).astype(int)
    amount_paid = np.where(
        trigger_event == 1, np.round(x["overdue_amount"] * rng.uniform(0.3, 1.5, n), 2), 0.0
    )
    repayment_strength = np.minimum(1.0, amount_paid / x["overdue_amount"])
    # Cron always charged the full overdue; event pilots tried some partials.
    charge_fraction = np.where(
        trigger_event == 1, rng.choice([1.0, 0.75, 0.5, 0.25], n, p=[0.55, 0.15, 0.2, 0.1]), 1.0
    )
    days_ago = rng.integers(1, 181, n)
    attempted = [today - timedelta(days=int(d)) for d in days_ago]
    payday = np.array([payday_window(d.day) for d in attempted])

    x.update(
        trigger_event=trigger_event,
        amount_paid=amount_paid,
        repayment_strength=np.round(repayment_strength, 3),
        charge_fraction=charge_fraction,
        payday_window=payday,
    )
    p = true_success_probability(x)
    success = rng.random(n) < p

    rows = []
    for i in range(n):
        decline = ""
        if not success[i]:
            weights = np.array([0.55 + 0.3 * (1 - x["pm_health"][i]), 0.25, 0.1, 0.1])
            decline = str(rng.choice(DECLINE_CODES, p=weights / weights.sum()))
        rows.append(
            {
                "attempt_id": f"att_{i:06d}",
                "attempted_at": attempted[i].isoformat(),
                "trigger": "event" if trigger_event[i] else "cron",
                "pm_health": float(x["pm_health"][i]),
                "recent_success": float(x["recent_success"][i]),
                "fails_7d": int(x["fails_7d"][i]),
                "days_overdue": int(x["days_overdue"][i]),
                "overdue_amount": float(x["overdue_amount"][i]),
                "amount_paid": float(amount_paid[i]),
                "repayment_strength": float(x["repayment_strength"][i]),
                "charge_fraction": float(charge_fraction[i]),
                "charge_amount": round(float(x["overdue_amount"][i] * charge_fraction[i]), 2),
                "trigger_event": int(trigger_event[i]),
                "payday_window": int(payday[i]),
                "success": int(success[i]),
                "decline_code": decline,
            }
        )
    return rows
