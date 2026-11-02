"""Backtest: cron vs always-charge-on-event vs smart-charge on fresh simulated customers.

Outcomes come from the hidden world simulator, with common random numbers so every
strategy faces the same luck per customer.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from app.config import Settings
from app.model import TrainedModel
from app.policy import choose_many
from app.schemas import BacktestResponse, StrategyResult
from app.world import payday_window, sample_customers, true_success_probability


def _result(name: str, attempted: np.ndarray, amount: np.ndarray, success: np.ndarray,
            settings: Settings) -> StrategyResult:
    attempts = int(attempted.sum())
    wins = int((attempted & success).sum())
    fails = attempts - wins
    recovered = float((amount * (attempted & success)).sum())
    fees = fails * settings.fail_fee
    return StrategyResult(
        strategy=name,
        attempts=attempts,
        successes=wins,
        failed_attempts=fails,
        success_rate=round(wins / attempts, 4) if attempts else 0.0,
        recovered=round(recovered, 2),
        fail_fees=round(fees, 2),
        net_recovered=round(recovered - fees, 2),
        recovered_per_attempt=round(recovered / attempts, 2) if attempts else 0.0,
    )


def run(model: TrainedModel, settings: Settings, n: int, seed: int) -> BacktestResponse:
    rng = np.random.default_rng(seed)
    x = sample_customers(n, rng)
    overdue = x["overdue_amount"]
    # Each customer just repaid something (the triggering event)
    amount_paid = np.round(overdue * rng.uniform(0.3, 1.5, n), 2)
    repay = np.minimum(1.0, amount_paid / overdue)
    days = [date.today() - timedelta(days=int(d)) for d in rng.integers(0, 30, n)]
    payday = np.array([payday_window(d.day) for d in days])
    luck = rng.random(n)  # common random numbers

    def outcome(trigger: int, fraction: np.ndarray, repayment: np.ndarray) -> np.ndarray:
        p = true_success_probability({
            **x,
            "repayment_strength": repayment,
            "charge_fraction": fraction,
            "trigger_event": np.full(n, trigger),
            "payday_window": payday,
        })
        return luck < p

    everyone = np.ones(n, dtype=bool)
    ones = np.ones(n)

    # Cron: tonight's batch, full overdue, no repayment context (activity signal lost)
    cron = _result("cron_nightly", everyone, overdue, outcome(0, ones, np.zeros(n)), settings)
    # Naive: charge full overdue on every repayment event
    naive = _result("always_charge_on_event", everyone, overdue, outcome(1, ones, repay), settings)

    bases = [
        {
            "pm_health": x["pm_health"][i],
            "recent_success": x["recent_success"][i],
            "fails_7d": x["fails_7d"][i],
            "days_overdue": x["days_overdue"][i],
            "repayment_strength": repay[i],
            "trigger_event": 1,
            "payday_window": payday[i],
        }
        for i in range(n)
    ]
    choices, _ = choose_many(model, bases, overdue, settings)
    gated = x["fails_7d"] >= settings.max_fails_7d
    attempted = np.array([c.action != "skip" for c in choices]) & ~gated
    fraction = np.array([c.fraction if c.action != "skip" else 1.0 for c in choices])
    smart_amount = np.round(overdue * fraction, 2)
    smart = _result("smart_charge", attempted, smart_amount, outcome(1, fraction, repay), settings)

    fail_cut = 1 - smart.failed_attempts / naive.failed_attempts if naive.failed_attempts else 0
    summary = (
        f"Smart vs always-charge: {fail_cut:.0%} fewer failed attempts, "
        f"success rate {smart.success_rate:.0%} vs {naive.success_rate:.0%}, "
        f"recovered SAR {smart.recovered:,.0f} vs {naive.recovered:,.0f} "
        f"(cron: SAR {cron.recovered:,.0f}, {cron.success_rate:.0%} success)."
    )
    return BacktestResponse(
        customers=n,
        total_overdue=round(float(overdue.sum()), 2),
        strategies=[cron, naive, smart],
        summary=summary,
        note="Synthetic customers; outcomes from the simulator, not real Tamara data.",
    )
