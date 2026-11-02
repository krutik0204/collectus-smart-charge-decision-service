"""Regenerate data/historical_charges.csv (synthetic autocharge history)."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.model import ensure_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/historical_charges.csv")
    args = parser.parse_args()
    out = Path(args.out)
    out.unlink(missing_ok=True)
    ensure_history(out, args.rows, args.seed)
    print(f"Wrote {args.rows} rows to {out}")
