from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import AsyncIterator

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import backtest, backtest_month, policy
from app.config import get_settings
from app.model import TrainedModel, ensure_history, load_or_train, train
from app.schemas import (
    BacktestRequest,
    BacktestResponse,
    DecisionAction,
    DecisionRequest,
    DecisionResponse,
    ModelInfo,
    MonthBacktestRequest,
    MonthBacktestResponse,
)
from app.world import FEATURES, card_features, card_label, payday_window

settings = get_settings()
state: dict[str, object] = {"model": None, "decisions": {}}


def _model() -> TrainedModel:
    return state["model"]  # type: ignore[return-value]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    state["model"] = load_or_train(
        Path(settings.model_path),
        Path(settings.history_path),
        settings.history_rows,
        settings.history_seed,
    )
    yield


app = FastAPI(
    title=settings.app_name,
    description="Collectus — AI decision gate for Smart Autocharge",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    model = _model()
    return {
        "status": "ok",
        "service": settings.app_name,
        "model_version": model.version if model else None,
        "auc": model.metrics.get("auc") if model else None,
    }


@app.post("/v1/decide", response_model=DecisionResponse)
def decide(body: DecisionRequest) -> DecisionResponse:
    decisions: dict[str, DecisionResponse] = state["decisions"]  # type: ignore[assignment]
    if body.event_id in decisions:
        return decisions[body.event_id].model_copy(update={"idempotent_replay": True})

    model = _model()
    f = body.features
    as_of = body.as_of or date.today()
    base = {
        "pm_health": f.pm_health,
        "recent_success": f.recent_success,
        "fails_7d": f.fails_7d,
        "days_overdue": f.days_overdue,
        "repayment_strength": f.repayment_strength,
        "trigger_event": 1,
        "payday_window": payday_window(as_of.day),
        **card_features(f.card_network, f.card_token, f.card_funding),
    }
    card = card_label(f.card_network, f.card_token, f.card_funding)

    declines_left = f.declines_left
    min_prob, harm = policy.budget_limits(declines_left, settings)
    gate = policy.hard_gate(body.overdue_amount, f.fails_7d, settings, declines_left)
    schedule_mode = body.event_type.startswith("schedule")
    if schedule_mode:
        base.update(trigger_event=0, repayment_strength=0.0)
    if gate:
        response = DecisionResponse(
            action=DecisionAction.SKIP,
            amount=0.0,
            score=0.0,
            reason=f"Hard gate: {gate} — skip",
            gate=gate,
            next_best_action=policy.next_best_action(base, np.zeros(1), declines_left),
            model_version=model.version,
            declines_left=declines_left,
            card=card,
        )
    elif schedule_mode or settings.repayment_policy == "full_gate":
        # Charge the full overdue only if it's worth spending a decline on.
        p_arr, bar_arr, go_arr = policy.full_charge_gate(
            model, [base], [declines_left], settings, scheduled=schedule_mode
        )
        p, bar, go = float(p_arr[0]), float(bar_arr[0]), bool(go_arr[0])
        drivers = model.drivers(policy.feature_row(base, 1.0))
        why = ", ".join(f"{d['direction']}{d['label']}" for d in drivers)
        moment = f"{'Scheduled attempt' if schedule_mode else 'Repayment moment'} on a {card}"
        budget = (
            f" Decline budget: {declines_left} left this month, so the bar is {bar:.0%}."
            if declines_left is not None and declines_left <= 1
            else ""
        )
        if go:
            verdict = f"charge the full {body.currency} {body.overdue_amount:.2f} now"
        else:
            verdict = "skip and keep the decline for a better moment"
        response = DecisionResponse(
            action=DecisionAction.CHARGE if go else DecisionAction.SKIP,
            amount=round(body.overdue_amount, 2) if go else 0.0,
            score=round(p, 4),
            reason=(
                f"{moment}: P(success) {p:.0%} {'clears' if go else 'is below'} the "
                f"{bar:.0%} bar — {verdict}.{budget} Drivers: {why}"
            ),
            next_best_action=(
                None if go else policy.next_best_action(base, np.array([p]), declines_left)
            ),
            drivers=drivers,
            candidates=(
                []
                if schedule_mode
                else policy.summarize_candidates(
                    policy.choose_many(model, [base], np.array([body.overdue_amount]), settings)[1][0],
                    body.overdue_amount,
                    settings,
                )
            ),
            model_version=model.version,
            declines_left=declines_left,
            min_success_prob_applied=bar,
            card=card,
        )
    else:
        overdue = np.array([body.overdue_amount])
        choices, probs = policy.choose_many(model, [base], overdue, settings, [declines_left])
        choice, probs_row = choices[0], probs[0]
        explained = policy.feature_row(base, choice.fraction or 1.0)
        drivers = model.drivers(explained)
        reason = policy.describe(choice, probs_row, body.overdue_amount, body.currency, drivers)
        if min_prob != settings.min_success_prob:
            reason += (
                f". Decline budget: {declines_left} left this month — "
                f"bar raised to {min_prob:.0%}"
            )
        response = DecisionResponse(
            action=DecisionAction(choice.action),
            amount=round(choice.amount, 2),
            score=round(choice.probability, 4),
            reason=reason,
            expected_value=round(choice.expected_value, 2),
            next_best_action=(
                policy.next_best_action(base, probs_row, declines_left)
                if choice.action == "skip"
                else None
            ),
            drivers=drivers,
            candidates=policy.summarize_candidates(
                probs_row, body.overdue_amount, settings.model_copy(update={"harm_weight": harm})
            ),
            model_version=model.version,
            declines_left=declines_left,
            min_success_prob_applied=min_prob,
            card=card,
        )

    decisions[body.event_id] = response
    return response


@app.post("/v1/backtest", response_model=BacktestResponse)
def run_backtest(body: BacktestRequest) -> BacktestResponse:
    policy_settings = settings
    if body.harm_weight is not None:
        policy_settings = settings.model_copy(update={"harm_weight": body.harm_weight})
    return backtest.run(_model(), policy_settings, body.n, body.seed)


@app.post("/v1/backtest/month", response_model=MonthBacktestResponse)
def run_month_backtest(body: MonthBacktestRequest) -> MonthBacktestResponse:
    policy_settings = settings
    if body.harm_weight is not None:
        policy_settings = settings.model_copy(update={"harm_weight": body.harm_weight})
    return backtest_month.run(
        _model(), policy_settings, body.n, body.seed, body.decline_budget, body.decline_budget_visa_mc
    )


@app.get("/v1/model", response_model=ModelInfo)
def model_info() -> ModelInfo:
    model = _model()
    return ModelInfo(
        version=model.version,
        features=FEATURES,
        coefficients=model.coefficients(),
        metrics=model.metrics,
        policy={
            "min_success_prob": settings.min_success_prob,
            "fail_fee": settings.fail_fee,
            "harm_weight": settings.harm_weight,
            "min_charge": settings.min_charge,
            "max_charge": settings.max_charge,
            "max_fails_7d": settings.max_fails_7d,
            "candidate_fractions": list(policy.CANDIDATE_FRACTIONS),
        },
    )


@app.post("/v1/model/retrain", response_model=ModelInfo)
def retrain() -> ModelInfo:
    history = ensure_history(Path(settings.history_path), settings.history_rows, settings.history_seed)
    model = train(history)
    joblib.dump(model, settings.model_path)
    state["model"] = model
    return model_info()
