from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "smart-charge-decision-service"
    app_env: str = "development"
    cors_origins: str = "*"

    history_path: str = "./data/historical_charges.csv"
    history_rows: int = 20000
    history_seed: int = 42
    model_path: str = "./data/model.joblib"

    # Policy — human-set limits that always sit above the model
    min_success_prob: float = 0.45
    fail_fee: float = 1.5  # PSP/issuer cost per failed attempt (SAR)
    harm_weight: float = 1.5  # customer friction + PM damage per SAR of failed charge
    min_charge: float = 10.0
    max_charge: float = 5000.0
    max_fails_7d: int = 5

    # Decline budget (schemes allow few declines per month): higher bar as slots run out
    budget_min_prob_2_left: float = 0.60
    budget_harm_multiplier_2_left: float = 1.5
    budget_min_prob_1_left: float = 0.75
    budget_harm_multiplier_1_left: float = 2.0

    # Repayment moments: "full_gate" (charge full overdue if P clears the bar — best in the
    # 30-day backtest) or "expected_value" (pick 100/75/50/25% by EV — partials)
    repayment_policy: str = "full_gate"
    repayment_min_prob: float = 0.10
    repayment_min_prob_last_decline: float = 0.30  # sweet spot in the 30-day backtest

    # Schedule mode: AI approves or blocks a scheduled (cron) attempt for the full overdue
    schedule_min_prob: float = 0.10
    schedule_min_prob_last_decline: float = 0.50

    # Plenty of declines left (e.g. a Visa/Mastercard card early in the month): try more freely
    plenty_declines_from: int = 3
    plenty_min_prob: float = 0.05

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
