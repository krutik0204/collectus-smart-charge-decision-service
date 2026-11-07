"""30-day backtest: one month per customer, with scheme decline budgets.

Each simulated customer starts the month with an overdue balance. During the month:
- Some customers make repayments (activity events) on random days.
- Cron runs on a fixed schedule (D0, D3, D18, D29), always charging the full remaining overdue.
- Every failed attempt uses one of the month's scheme declines (Mada 3, Visa/Mastercard 6);
  with none left, the card can't be charged. After a decline, no more attempts that day.
- Failed attempts count towards fails_7d for 7 days.

Strategies run on the same customers and the same luck (common random numbers), and
outcomes come from the hidden world simulator — synthetic, not real Tamara data.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from app.config import Settings
from app.model import TrainedModel
from app.policy import choose_many, full_charge_gate
from app.schemas import MonthBacktestResponse, MonthStrategyResult
from app.world import payday_window, sample_customers, true_success_probability

CRON_DAYS = (0, 3, 18, 29)
RULE_MIN_PM_HEALTH = 0.5
RULE_MAX_FAILS_7D = 3

# A policy returns, for the customers in `mask`, the fraction of overdue to charge (0 = no attempt).
Policy = Callable[[np.ndarray, dict], np.ndarray]


class _World:
    def __init__(self, n: int, seed: int, days: int) -> None:
        rng = np.random.default_rng(seed)
        self.n, self.days = n, days
        self.x = sample_customers(n, rng, np.random.default_rng(seed + 1))
        self.overdue0 = self.x["overdue_amount"]
        events = rng.poisson(1.0, n)  # repayment events this month
        self.event_day = np.zeros((n, days), dtype=bool)
        for i in range(n):
            k = min(int(events[i]), days)
            if k:
                self.event_day[i, rng.choice(days, size=k, replace=False)] = True
        self.repay_strength = rng.uniform(0.3, 1.0, (n, days))
        self.luck_event = rng.random((n, days))
        self.luck_cron = rng.random((n, days))
        # Month starts on the 1st, so days 0-2 and 26-29 fall in the payday window.
        self.payday = np.array([payday_window(d + 1) for d in range(days)])


def _simulate(world: _World, name: str, budget: np.ndarray, on_event: Optional[Policy],
              on_cron: Optional[Policy], settings: Settings) -> MonthStrategyResult:
    n, days, x = world.n, world.days, world.x
    remaining = world.overdue0.copy()
    declines_left = budget.copy()
    fail_hist = np.zeros((n, days), dtype=bool)
    first_collect = np.full(n, -1)
    attempts = successes = 0
    recovered = 0.0

    def state(d: int) -> dict:
        recent = fail_hist[:, max(0, d - 7):d].sum(axis=1)
        return {
            "fails_7d": np.minimum(x["fails_7d"] + recent, 8),
            "days_overdue": x["days_overdue"] + d,
            "declines_left": declines_left,
            "remaining": remaining,
            "payday": int(world.payday[d]),
        }

    def attempt(mask: np.ndarray, fraction: np.ndarray, trigger: int, repay: np.ndarray,
                luck: np.ndarray, d: int) -> np.ndarray:
        nonlocal attempts, successes, recovered
        s = state(d)
        p = true_success_probability({
            **x,
            "fails_7d": s["fails_7d"],
            "days_overdue": s["days_overdue"],
            "repayment_strength": repay,
            "charge_fraction": np.where(fraction > 0, fraction, 1.0),
            "trigger_event": np.full(n, trigger),
            "payday_window": np.full(n, s["payday"]),
        })
        tried = mask & (fraction > 0)
        won = tried & (luck < p)
        lost = tried & ~won
        amount = np.round(remaining * fraction, 2)
        recovered += float(amount[won].sum())
        remaining[won] = np.round(remaining[won] - amount[won], 2)
        first_collect[won & (first_collect < 0)] = d
        declines_left[lost] -= 1
        fail_hist[lost, d] = True
        attempts += int(tried.sum())
        successes += int(won.sum())
        return lost

    for d in range(days):
        open_ = (remaining > 0.01) & (declines_left > 0)
        failed_today = np.zeros(n, dtype=bool)
        if on_event is not None:
            mask = open_ & world.event_day[:, d]
            if mask.any():
                frac = on_event(mask, {**state(d), "repay": world.repay_strength[:, d], "d": d})
                failed_today = attempt(mask, frac, 1, world.repay_strength[:, d],
                                       world.luck_event[:, d], d)
        if on_cron is not None and d in CRON_DAYS:
            mask = (remaining > 0.01) & (declines_left > 0) & ~failed_today
            if mask.any():
                frac = on_cron(mask, {**state(d), "repay": np.zeros(n), "d": d})
                attempt(mask, frac, 0, np.zeros(n), world.luck_cron[:, d], d)

    total = float(world.overdue0.sum())
    failed = attempts - successes
    cleared = remaining <= 0.01
    collected = first_collect >= 0
    return MonthStrategyResult(
        strategy=name,
        attempts=attempts,
        failed_attempts=failed,
        success_rate=round(successes / attempts, 4) if attempts else 0.0,
        recovered=round(recovered, 2),
        recovery_rate=round(recovered / total, 4) if total else 0.0,
        fail_fees=round(failed * settings.fail_fee, 2),
        customers_fully_cleared=round(float(cleared.mean()), 4),
        customers_out_of_declines=round(float((declines_left <= 0).mean()), 4),
        avg_day_first_collection=(
            round(float(first_collect[collected].mean()), 1) if collected.any() else None
        ),
    )


def run(model: TrainedModel, settings: Settings, n: int, seed: int, budget: int,
        budget_visa_mc: int = 6, days: int = 30) -> MonthBacktestResponse:
    world = _World(n, seed, days)
    x = world.x
    budgets = np.where(x["card_network"] == "mada", budget, budget_visa_mc)

    def full(mask: np.ndarray, _: dict) -> np.ndarray:
        return np.where(mask, 1.0, 0.0)

    def rule(mask: np.ndarray, s: dict) -> np.ndarray:
        ok = (x["pm_health"] > RULE_MIN_PM_HEALTH) & (s["fails_7d"] < RULE_MAX_FAILS_7D)
        return np.where(mask & ok, 1.0, 0.0)

    def smart(trigger: int) -> Policy:
        def decide(mask: np.ndarray, s: dict) -> np.ndarray:
            idx = np.flatnonzero(mask & (s["fails_7d"] < settings.max_fails_7d))
            out = np.zeros(world.n)
            if idx.size == 0:
                return out
            bases = [
                {
                    "pm_health": x["pm_health"][i],
                    "recent_success": x["recent_success"][i],
                    "fails_7d": s["fails_7d"][i],
                    "days_overdue": s["days_overdue"][i],
                    "repayment_strength": s["repay"][i] if trigger else 0.0,
                    "trigger_event": trigger,
                    "payday_window": s["payday"],
                    "network_visa": x["network_visa"][i],
                    "network_mastercard": x["network_mastercard"][i],
                    "apple_pay": x["apple_pay"][i],
                    "funding_credit": x["funding_credit"][i],
                    "funding_prepaid": x["funding_prepaid"][i],
                }
                for i in idx
            ]
            choices, _ = choose_many(model, bases, s["remaining"][idx], settings,
                                     [int(s["declines_left"][i]) for i in idx])
            out[idx] = [c.fraction if c.action != "skip" else 0.0 for c in choices]
            return out
        return decide

    def ai_full_gate(scheduled: bool) -> Policy:
        """Same as /v1/decide: charge the full overdue only if P clears the bar."""
        def decide(mask: np.ndarray, s: dict) -> np.ndarray:
            idx = np.flatnonzero(mask & (s["fails_7d"] < settings.max_fails_7d))
            out = np.zeros(world.n)
            if idx.size == 0:
                return out
            bases = [
                {
                    "pm_health": x["pm_health"][i],
                    "recent_success": x["recent_success"][i],
                    "fails_7d": s["fails_7d"][i],
                    "days_overdue": s["days_overdue"][i],
                    "repayment_strength": 0.0 if scheduled else s["repay"][i],
                    "trigger_event": 0 if scheduled else 1,
                    "payday_window": s["payday"],
                    "network_visa": x["network_visa"][i],
                    "network_mastercard": x["network_mastercard"][i],
                    "apple_pay": x["apple_pay"][i],
                    "funding_credit": x["funding_credit"][i],
                    "funding_prepaid": x["funding_prepaid"][i],
                }
                for i in idx
            ]
            _, _, go = full_charge_gate(model, bases, [int(s["declines_left"][i]) for i in idx],
                                        settings, scheduled)
            out[idx] = np.where(go, 1.0, 0.0)
            return out
        return decide

    strategies = [
        _simulate(world, "cron_only", budgets, None, full, settings),
        _simulate(world, "cron_plus_always_charge_on_repayment", budgets, full, full, settings),
        _simulate(world, "cron_plus_simple_rule_on_repayment", budgets, rule, full, settings),
        _simulate(world, "cron_plus_smart_partials_on_repayment", budgets, smart(1), full, settings),
        _simulate(world, "charge_on_repayment_plus_ai_gated_cron", budgets, ai_full_gate(False),
                  ai_full_gate(True), settings),
    ]
    by = {s.strategy: s for s in strategies}
    cron, naive = by["cron_only"], by["cron_plus_always_charge_on_repayment"]
    ai = by["charge_on_repayment_plus_ai_gated_cron"]

    def pct(a: float, b: float) -> str:
        return f"{(a / b - 1):+.0%}" if b else "n/a"

    summary = (
        f"Over {days} days ({budget} declines per Mada card, {budget_visa_mc} per Visa/Mastercard), "
        f"charging on repayment with an "
        f"AI-gated cron recovers {ai.recovery_rate:.1%} of overdue vs {cron.recovery_rate:.1%} for "
        f"cron only ({pct(ai.recovered, cron.recovered)}) with "
        f"{pct(ai.failed_attempts, cron.failed_attempts)} failed attempts; vs always-charge "
        f"({naive.recovery_rate:.1%}) it makes {pct(ai.failed_attempts, naive.failed_attempts)} "
        f"failed attempts. Customers out of declines: {ai.customers_out_of_declines:.0%} vs "
        f"{naive.customers_out_of_declines:.0%}."
    )
    return MonthBacktestResponse(
        customers=n,
        days=days,
        decline_budget=budget,
        decline_budget_visa_mc=budget_visa_mc,
        total_overdue=round(float(world.overdue0.sum()), 2),
        strategies=strategies,
        summary=summary,
        note="Synthetic customers; outcomes from the simulator, not real Tamara data.",
    )
