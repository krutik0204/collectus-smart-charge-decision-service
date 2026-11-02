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

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
