from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import AsyncIterator

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import backtest, policy
from app.config import get_settings
from app.model import TrainedModel, ensure_history, load_or_train, train
from app.schemas import (
    BacktestRequest,
    BacktestResponse,
    DecisionAction,
    DecisionRequest,
    DecisionResponse,
    ModelInfo,
)
from app.world import FEATURES, payday_window

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
    }

    gate = policy.hard_gate(body.overdue_amount, f.fails_7d, settings)
    if gate:
        response = DecisionResponse(
            action=DecisionAction.SKIP,
            amount=0.0,
            score=0.0,
            reason=f"Hard gate: {gate} — skip",
            gate=gate,
            next_best_action=policy.next_best_action(base, np.zeros(1)),
            model_version=model.version,
        )
    else:
        overdue = np.array([body.overdue_amount])
        choices, probs = policy.choose_many(model, [base], overdue, settings)
        choice, probs_row = choices[0], probs[0]
        explained = policy.feature_row(base, choice.fraction or 1.0)
        drivers = model.drivers(explained)
        response = DecisionResponse(
            action=DecisionAction(choice.action),
            amount=round(choice.amount, 2),
            score=round(choice.probability, 4),
            reason=policy.describe(choice, probs_row, body.overdue_amount, body.currency, drivers),
            expected_value=round(choice.expected_value, 2),
            next_best_action=(
                policy.next_best_action(base, probs_row) if choice.action == "skip" else None
            ),
            drivers=drivers,
            candidates=policy.summarize_candidates(probs_row, body.overdue_amount, settings),
            model_version=model.version,
        )

    decisions[body.event_id] = response
    return response


@app.post("/v1/backtest", response_model=BacktestResponse)
def run_backtest(body: BacktestRequest) -> BacktestResponse:
    policy_settings = settings
    if body.harm_weight is not None:
        policy_settings = settings.model_copy(update={"harm_weight": body.harm_weight})
    return backtest.run(_model(), policy_settings, body.n, body.seed)


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
