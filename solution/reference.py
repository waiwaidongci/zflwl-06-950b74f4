#!/usr/bin/env python3
"""Reference solver for joint multi-run calibration structure recovery."""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


TOP_KEYS = {
    "schema",
    "repeats",
    "regime_count",
    "references",
    "field_units",
    "residual_tolerance",
}
UNIT_KEYS = {"id", "runs"}
REFERENCE_ID = re.compile(r"R-[A-Z0-9]{4}\Z")
FIELD_ID = re.compile(r"U-[A-Z0-9]{4}\Z")
MIN_SEGMENT = 6


class Refused(ValueError):
    pass


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refused(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise Refused(f"{label} must be finite")
    return number


def _integer(value: Any, label: str, allowed: set[int]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Refused(f"{label} must be an integer")
    if value not in allowed:
        raise Refused(f"{label} is outside its allowed values")
    return value


def _parse_units(
    raw_units: Any,
    pattern: re.Pattern[str],
    repeats: int,
    expected_length: int | None,
) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(raw_units, list) or not 6 <= len(raw_units) <= 9:
        raise Refused("unit collections must contain between 6 and 9 entries")
    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    length = expected_length
    for index, raw in enumerate(raw_units):
        if not isinstance(raw, dict) or set(raw) != UNIT_KEYS:
            raise Refused(f"unit {index} has an invalid key set")
        unit_id = raw.get("id")
        if not isinstance(unit_id, str) or pattern.fullmatch(unit_id) is None:
            raise Refused(f"unit {index} has an invalid id")
        if unit_id in seen:
            raise Refused("unit ids must be unique")
        seen.add(unit_id)
        raw_runs = raw.get("runs")
        if not isinstance(raw_runs, list) or len(raw_runs) != repeats:
            raise Refused(f"{unit_id} has an invalid run count")
        runs: list[list[float]] = []
        for run_index, raw_series in enumerate(raw_runs):
            if not isinstance(raw_series, list) or not 32 <= len(raw_series) <= 48:
                raise Refused(f"{unit_id} run {run_index} has an invalid length")
            if length is None:
                length = len(raw_series)
            if len(raw_series) != length:
                raise Refused("all series must have the same length")
            series = []
            for value in raw_series:
                number = _finite_number(value, f"{unit_id} series value")
                if not -100.0 <= number <= 100.0:
                    raise Refused(f"{unit_id} series value is outside its range")
                series.append(number)
            runs.append(series)
        units.append({"id": unit_id, "runs": runs})
    return units, length or 0


def parse_input(
    path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    int,
    float,
]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Refused("input cannot be read") from exc
    except json.JSONDecodeError as exc:
        raise Refused("input is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != TOP_KEYS:
        raise Refused("top-level keys do not match repeat-calibration-v2")
    if payload.get("schema") != "repeat-calibration-v2":
        raise Refused("unsupported schema")
    repeats = _integer(payload.get("repeats"), "repeats", {3, 4})
    regime_count = _integer(payload.get("regime_count"), "regime_count", {3, 4})
    tolerance = _finite_number(
        payload.get("residual_tolerance"), "residual_tolerance"
    )
    if not 0.0001 <= tolerance <= 2.0:
        raise Refused("residual_tolerance is outside its allowed range")
    references, reference_length = _parse_units(
        payload.get("references"), REFERENCE_ID, repeats, None
    )
    fields, field_length = _parse_units(
        payload.get("field_units"), FIELD_ID, repeats, reference_length
    )
    if len(references) != len(fields) or reference_length != field_length:
        raise Refused("reference and field collections must have matching shapes")
    return references, fields, repeats, regime_count, tolerance


def _invert(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    work = [
        row[:] + [1.0 if row_index == column else 0.0 for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1e-13:
            raise Refused("segmented regression is singular")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        for item in range(column, size * 2):
            work[column][item] /= scale
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            for item in range(column, size * 2):
                work[row][item] -= factor * work[column][item]
    return [row[size:] for row in work]


def _design_row(value: float, index: int, breaks: tuple[int, ...]) -> list[float]:
    row = [value, 1.0]
    row.extend(1.0 if index >= break_index else 0.0 for break_index in breaks)
    return row


def _basis(
    reference: list[float],
    breaks: tuple[int, ...],
) -> list[list[float]]:
    width = 2 + len(breaks)
    normal = [[0.0 for _ in range(width)] for _ in range(width)]
    for index, predictor in enumerate(reference):
        row = _design_row(predictor, index, breaks)
        for left in range(width):
            for right in range(width):
                normal[left][right] += row[left] * row[right]
    return _invert(normal)


def _fit_with_basis(
    reference: list[float],
    field: list[float],
    breaks: tuple[int, ...],
    inverse: list[list[float]],
) -> tuple[float, list[float]]:
    width = 2 + len(breaks)
    cross = [0.0 for _ in range(width)]
    target_squares = []
    for index, (predictor, target) in enumerate(zip(reference, field)):
        row = _design_row(predictor, index, breaks)
        for item in range(width):
            cross[item] += row[item] * target
        target_squares.append(target * target)
    coefficients = [
        math.fsum(inverse[row][column] * cross[column] for column in range(width))
        for row in range(width)
    ]
    sse = math.fsum(target_squares) - math.fsum(
        coefficients[index] * cross[index] for index in range(width)
    )
    if sse < 0.0 and sse > -1e-8:
        sse = 0.0
    return sse, coefficients


def _hungarian(cost: list[list[float]]) -> tuple[list[int], float]:
    size = len(cost)
    u = [0.0 for _ in range(size + 1)]
    v = [0.0 for _ in range(size + 1)]
    p = [0 for _ in range(size + 1)]
    way = [0 for _ in range(size + 1)]
    for row in range(1, size + 1):
        p[0] = row
        column = 0
        minimum = [math.inf for _ in range(size + 1)]
        used = [False for _ in range(size + 1)]
        while True:
            used[column] = True
            current_row = p[column]
            delta = math.inf
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                current = (
                    cost[current_row - 1][candidate - 1]
                    - u[current_row]
                    - v[candidate]
                )
                if current < minimum[candidate]:
                    minimum[candidate] = current
                    way[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    u[p[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if p[column] == 0:
                break
        while True:
            previous = way[column]
            p[column] = p[previous]
            column = previous
            if column == 0:
                break
    assignment = [-1 for _ in range(size)]
    for column in range(1, size + 1):
        assignment[p[column] - 1] = column - 1
    total = math.fsum(cost[row][assignment[row]] for row in range(size))
    return assignment, total


def _break_vectors(sample_count: int, regime_count: int) -> list[tuple[int, ...]]:
    candidates: list[tuple[int, ...]] = []
    for values in itertools.combinations(
        range(MIN_SEGMENT, sample_count - MIN_SEGMENT + 1),
        regime_count - 1,
    ):
        edges = (0, *values, sample_count)
        if all(
            edges[index + 1] - edges[index] >= MIN_SEGMENT
            for index in range(len(edges) - 1)
        ):
            candidates.append(values)
    return candidates


def _best_run_order(
    fits: dict[tuple[int, int, int, int], float],
    field_index: int,
    reference_index: int,
    repeats: int,
) -> tuple[tuple[int, ...], float]:
    best_order: tuple[int, ...] | None = None
    best_sse = math.inf
    for order in itertools.permutations(range(repeats)):
        sse = math.fsum(
            fits[(field_index, local_run, reference_index, reference_run)]
            for local_run, reference_run in enumerate(order)
        )
        if sse < best_sse:
            best_sse = sse
            best_order = order
    if best_order is None:
        raise Refused("no run correspondence exists")
    return best_order, best_sse


def estimate(
    references: list[dict[str, Any]],
    fields: list[dict[str, Any]],
    repeats: int,
    regime_count: int,
) -> dict[str, Any]:
    sample_count = len(references[0]["runs"][0])
    break_vectors = _break_vectors(sample_count, regime_count)
    if not break_vectors:
        raise Refused("no admissible shared break vector exists")

    best: (
        tuple[
            float,
            tuple[int, ...],
            list[int],
            dict[tuple[int, int], tuple[int, ...]],
        ]
        | None
    ) = None
    for breaks in break_vectors:
        bases = {
            (reference_index, reference_run): _basis(reference_series, breaks)
            for reference_index, reference in enumerate(references)
            for reference_run, reference_series in enumerate(reference["runs"])
        }
        fits: dict[tuple[int, int, int, int], float] = {}
        for field_index, field in enumerate(fields):
            for local_run, field_series in enumerate(field["runs"]):
                for reference_index, reference in enumerate(references):
                    for reference_run, reference_series in enumerate(reference["runs"]):
                        sse, _ = _fit_with_basis(
                            reference_series,
                            field_series,
                            breaks,
                            bases[(reference_index, reference_run)],
                        )
                        fits[(field_index, local_run, reference_index, reference_run)] = sse

        pair_orders: dict[tuple[int, int], tuple[int, ...]] = {}
        pair_cost = [[0.0 for _ in references] for _ in fields]
        for field_index in range(len(fields)):
            for reference_index in range(len(references)):
                order, sse = _best_run_order(
                    fits, field_index, reference_index, repeats
                )
                pair_orders[(field_index, reference_index)] = order
                pair_cost[field_index][reference_index] = sse
        assignment, total_sse = _hungarian(pair_cost)
        if best is None or total_sse < best[0]:
            best = (total_sse, breaks, assignment, pair_orders)

    if best is None:
        raise Refused("joint calibration has no solution")
    total_sse, breaks, assignment, pair_orders = best
    matches: dict[str, str] = {}
    run_orders: dict[str, list[int]] = {}
    drift_steps: dict[str, list[list[float]]] = {}
    for field_index, field in enumerate(fields):
        reference_index = assignment[field_index]
        reference = references[reference_index]
        order = pair_orders[(field_index, reference_index)]
        matches[field["id"]] = reference["id"]
        run_orders[field["id"]] = list(order)
        local_drifts: list[list[float]] = []
        for local_run, reference_run in enumerate(order):
            inverse = _basis(reference["runs"][reference_run], breaks)
            _, coefficients = _fit_with_basis(
                reference["runs"][reference_run],
                field["runs"][local_run],
                breaks,
                inverse,
            )
            local_drifts.append([round(value, 6) for value in coefficients[2:]])
        drift_steps[field["id"]] = local_drifts

    denominator = len(fields) * repeats * sample_count
    residual_rms = math.sqrt(total_sse / denominator)
    return {
        "break_indices": list(breaks),
        "drift_steps": drift_steps,
        "matches": matches,
        "residual_rms": round(residual_rms, 6),
        "run_orders": run_orders,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("REFUSED: expected input and output paths", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    try:
        references, fields, repeats, regime_count, tolerance = parse_input(source)
        result = estimate(references, fields, repeats, regime_count)
        if float(result["residual_rms"]) > tolerance:
            raise Refused("joint residual exceeds the configured tolerance")
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    target.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
