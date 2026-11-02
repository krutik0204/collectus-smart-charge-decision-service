"""Success-probability model: calibrated logistic regression on historical attempts."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.world import FEATURES, generate_history

FEATURE_LABELS = {
    "pm_health": "payment method health",
    "recent_success": "recent payment success",
    "fails_7d": "failed attempts (7d)",
    "days_overdue": "days overdue",
    "repayment_strength": "repayment just made",
    "charge_fraction": "size of charge vs overdue",
    "trigger_event": "charging right after activity",
    "payday_window": "payday window",
}


@dataclass
class TrainedModel:
    pipeline: Pipeline
    version: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def predict(self, rows: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(rows)[:, 1]

    def drivers(self, row: np.ndarray, top: int = 3) -> list[dict[str, Any]]:
        """Per-feature contribution to the log-odds (coef × standardized value)."""
        scaler: StandardScaler = self.pipeline.named_steps["scale"]
        clf: LogisticRegression = self.pipeline.named_steps["clf"]
        scaled = scaler.transform(row.reshape(1, -1))[0]
        contrib = clf.coef_[0] * scaled
        order = np.argsort(-np.abs(contrib))
        out = []
        for idx in order:
            name = FEATURES[idx]
            if name == "trigger_event":
                continue  # constant on the decide path; not informative per customer
            out.append(
                {
                    "feature": name,
                    "label": FEATURE_LABELS[name],
                    "value": float(row[idx]),
                    "impact": round(float(contrib[idx]), 3),
                    "direction": "+" if contrib[idx] >= 0 else "-",
                }
            )
            if len(out) == top:
                break
        return out

    def coefficients(self) -> dict[str, float]:
        clf: LogisticRegression = self.pipeline.named_steps["clf"]
        return {f: round(float(c), 4) for f, c in zip(FEATURES, clf.coef_[0])}


def ensure_history(path: Path, rows: int, seed: int) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    history = generate_history(rows, seed=seed)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    return path


def load_history(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open() as fh:
        reader = list(csv.DictReader(fh))
    X = np.array([[float(r[f]) for f in FEATURES] for r in reader])
    y = np.array([int(r["success"]) for r in reader])
    trigger = np.array([r["trigger"] for r in reader])
    return X, y, trigger


def train(history_path: Path, seed: int = 7) -> TrainedModel:
    X, y, trigger = load_history(history_path)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    pipeline = Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
    )
    pipeline.fit(X_tr, y_tr)
    p = pipeline.predict_proba(X_te)[:, 1]

    metrics = {
        "rows": int(len(y)),
        "train_rows": int(len(y_tr)),
        "test_rows": int(len(y_te)),
        "auc": round(float(roc_auc_score(y_te, p)), 4),
        "brier": round(float(brier_score_loss(y_te, p)), 4),
        "log_loss": round(float(log_loss(y_te, p)), 4),
        # Business-case stats straight from history
        "history_success_rate": round(float(y.mean()), 4),
        "cron_success_rate": round(float(y[trigger == "cron"].mean()), 4),
        "event_success_rate": round(float(y[trigger == "event"].mean()), 4),
    }
    version = "logreg-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return TrainedModel(pipeline=pipeline, version=version, metrics=metrics)


def load_or_train(model_path: Path, history_path: Path, rows: int, seed: int) -> TrainedModel:
    if model_path.exists():
        return joblib.load(model_path)
    ensure_history(history_path, rows, seed)
    model = train(history_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model
