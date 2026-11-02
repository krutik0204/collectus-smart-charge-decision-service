from fastapi.testclient import TestClient

from app.main import app


def _body(event_id, pm_health, fails_7d, recent_success, overdue, paid=None, as_of="2026-11-02"):
    paid = overdue if paid is None else paid
    return {
        "event_id": event_id,
        "customer_id": "c_test",
        "amount_paid": paid,
        "overdue_amount": overdue,
        "as_of": as_of,
        "features": {
            "days_overdue": 15,
            "pm_health": pm_health,
            "fails_7d": fails_7d,
            "recent_success": recent_success,
            "repayment_strength": min(1.0, paid / overdue) if overdue else 1.0,
        },
    }


def test_demo_personas():
    with TestClient(app) as client:
        aisha = client.post("/v1/decide", json=_body("t_aisha", 0.9, 0, 0.85, 45.0)).json()
        omar = client.post("/v1/decide", json=_body("t_omar", 0.25, 4, 0.15, 130.0)).json()
        layla = client.post("/v1/decide", json=_body("t_layla", 0.6, 1, 0.55, 249.75)).json()
    assert aisha["action"] == "charge" and aisha["amount"] == 45.0
    assert omar["action"] == "skip" and omar["amount"] == 0
    assert layla["action"] == "partial" and 0 < layla["amount"] < 249.75


def test_hard_gates():
    with TestClient(app) as client:
        no_overdue = client.post("/v1/decide", json=_body("t_none", 0.9, 0, 0.9, 0.0, paid=100)).json()
        capped = client.post("/v1/decide", json=_body("t_cap", 0.9, 6, 0.9, 100.0)).json()
    assert no_overdue["action"] == "skip" and no_overdue["gate"]
    assert capped["action"] == "skip" and "Attempt cap" in capped["gate"]


def test_idempotent_replay():
    with TestClient(app) as client:
        first = client.post("/v1/decide", json=_body("t_same", 0.9, 0, 0.85, 45.0)).json()
        again = client.post("/v1/decide", json=_body("t_same", 0.1, 4, 0.1, 999.0)).json()
    assert again["idempotent_replay"] is True
    assert again["action"] == first["action"] and again["amount"] == first["amount"]


def test_backtest_smart_fails_less():
    with TestClient(app) as client:
        result = client.post("/v1/backtest", json={"n": 500}).json()
    by_name = {s["strategy"]: s for s in result["strategies"]}
    assert by_name["smart_charge"]["failed_attempts"] < by_name["always_charge_on_event"]["failed_attempts"]
    assert by_name["smart_charge"]["success_rate"] > by_name["cron_nightly"]["success_rate"]
