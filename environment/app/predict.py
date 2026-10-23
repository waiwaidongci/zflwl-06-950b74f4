#!/usr/bin/env python3
"""Starter entry point for repeat calibration archives."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("REFUSED: expected input and output paths", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("REFUSED: unreadable input", file=sys.stderr)
        return 2

    repeats = payload.get("repeats", 3)
    regime_count = payload.get("regime_count", 3)
    fields = payload.get("field_units", [])
    references = payload.get("references", [])
    matches = {}
    run_orders = {}
    drift_steps = {}
    for field, reference in zip(fields, references):
        field_id = field["id"]
        matches[field_id] = reference["id"]
        run_orders[field_id] = [0 for _ in range(repeats)]
        drift_steps[field_id] = [
            [0.0 for _ in range(regime_count - 1)]
            for _ in range(repeats)
        ]
    result = {
        "break_indices": [6, 20][: regime_count - 1],
        "drift_steps": drift_steps,
        "matches": matches,
        "residual_rms": 0.0,
        "run_orders": run_orders,
    }
    target.write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
