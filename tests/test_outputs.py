from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest


ENTRY = os.environ.get("REPEAT_CALIBRATION_ENTRY", "/app/predict.py")

SAFE_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}
REFERENCE_IDS = [
    "R-A7K2",
    "R-M4Q9",
    "R-T1B8",
    "R-C6N3",
    "R-W9F5",
    "R-L2D7",
    "R-H8P1",
    "R-V3S6",
    "R-E5J4",
]
FIELD_IDS = [
    "U-B9R3",
    "U-N2W7",
    "U-G6C1",
    "U-Q4M8",
    "U-Z8L5",
    "U-F1T9",
    "U-K7D2",
    "U-P3H6",
    "U-Y5A4",
]


def _run(source: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-S", "-P", "-B", ENTRY, str(source), str(output)],
        text=True,
        capture_output=True,
        env=SAFE_ENV,
        timeout=60,
        check=False,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _field_assignment(count: int) -> list[int]:
    return [(5 * index + 3) % count for index in range(count)]


def _reference_run_order(field_index: int, repeats: int) -> tuple[int, ...]:
    if repeats == 3:
        return tuple((field_index + offset) % 3 for offset in (1, 2, 0))
    return tuple((field_index + offset) % 4 for offset in (1, 3, 2, 0))


def _payload(
    *,
    count: int = 8,
    repeats: int = 4,
    sample_count: int = 36,
    regime_count: int = 3,
    breaks: tuple[int, ...] | None = None,
    noise: float = 0.0,
    tolerance: float = 0.05,
) -> tuple[dict[str, object], dict[str, object]]:
    if breaks is None:
        if regime_count == 3:
            breaks = (11, 25) if sample_count == 36 else (9, 23)
        else:
            breaks = (9, 19, 31) if sample_count == 40 else (7, 17, 27)
    times = [index / (sample_count - 1) for index in range(sample_count)]
    references: list[dict[str, object]] = []
    for station, reference_id in enumerate(REFERENCE_IDS[:count]):
        phase = 0.37 * station
        frequency = 1.03 + 0.083 * station
        runs: list[list[float]] = []
        for run_index in range(repeats):
            series = [
                17.5
                + 0.42 * station
                + (4.1 + 0.19 * run_index + 0.07 * station)
                * math.sin(
                    2.0 * math.pi * frequency * time
                    + phase
                    + 0.16 * run_index
                )
                + (0.9 + 0.04 * station)
                * math.cos(
                    2.0
                    * math.pi
                    * (0.29 + 0.037 * station)
                    * time
                    + 0.11 * run_index
                )
                for time in times
            ]
            runs.append(series)
        references.append({"id": reference_id, "runs": runs})

    assignment = _field_assignment(count)
    fields: list[dict[str, object]] = []
    matches: dict[str, str] = {}
    run_orders: dict[str, list[int]] = {}
    drift_steps: dict[str, list[list[float]]] = {}
    for field_index, field_id in enumerate(FIELD_IDS[:count]):
        station = assignment[field_index]
        order = _reference_run_order(field_index, repeats)
        matches[field_id] = REFERENCE_IDS[station]
        run_orders[field_id] = list(order)
        local_drifts: list[list[float]] = []
        runs = []
        for local_run, reference_run in enumerate(order):
            scale = 0.75 + 0.021 * field_index + 0.013 * local_run
            offset = -3.4 + 0.18 * field_index - 0.31 * local_run
            drifts = [
                0.85
                + 0.21 * (field_index + 1)
                - 0.07 * local_run
                + 0.08 * reference_run,
                -1.35
                + 0.13 * (field_index + 1)
                + 0.05 * local_run
                - 0.11 * reference_run,
                0.48
                + 0.09 * (field_index + 1)
                - 0.04 * local_run
                + 0.06 * reference_run,
            ][: regime_count - 1]
            local_drifts.append(drifts)
            series = []
            for sample_index, value in enumerate(
                references[station]["runs"][reference_run]
            ):
                level = scale * value + offset
                level += sum(
                    drifts[index]
                    for index, break_index in enumerate(breaks)
                    if sample_index >= break_index
                )
                level += noise * math.sin(
                    (sample_index + 1)
                    * (field_index + 2 + local_run * 2)
                )
                series.append(level)
            runs.append(series)
        fields.append({"id": field_id, "runs": runs})
        drift_steps[field_id] = local_drifts

    payload: dict[str, object] = {
        "schema": "repeat-calibration-v2",
        "repeats": repeats,
        "regime_count": regime_count,
        "references": references,
        "field_units": fields,
        "residual_tolerance": tolerance,
    }
    expected = {
        "break_indices": list(breaks),
        "drift_steps": drift_steps,
        "matches": matches,
        "run_orders": run_orders,
    }
    return payload, expected


def test_joint_structure_recovers_matches_run_orders_and_breaks(tmp_path: Path) -> None:
    """题面：
    「输出中的所有浮点数保留六位小数。`break_indices` 保持整数，`matches`、`run_orders`」
    """

    # REPEAT-CONTEXT REPEAT-CONTEXT-02 REPEAT-LATENT REPEAT-LATENT-02
    # REPEAT-LOCAL-SHAPE REPEAT-UNIQUE
    payload, expected = _payload(noise=0.003)
    source = tmp_path / "joint.json"
    output = tmp_path / "joint-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["matches"] == expected["matches"]
    assert result["run_orders"] == expected["run_orders"]
    assert result["break_indices"] == expected["break_indices"]
    assert result["residual_rms"] <= payload["residual_tolerance"]


def test_alternate_break_vector_is_selected_globally(tmp_path: Path) -> None:
    """题面：
    「`residual_tolerance` 六项。`schema` 必须是 `"repeat-calibration-v2"`。」
    """

    # REPEAT-SHAREDBREAK
    payload, expected = _payload(
        count=7,
        repeats=3,
        sample_count=40,
        breaks=(8, 21),
        noise=0.002,
    )
    source = tmp_path / "alternate-break.json"
    output = tmp_path / "alternate-break-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["break_indices"] == expected["break_indices"]
    assert result["matches"] == expected["matches"]


def test_four_regime_archive_recovers_three_shared_breaks(tmp_path: Path) -> None:
    """题面：
    「顶层恰好包含 `schema`、`repeats`、`regime_count`、`references`、`field_units`、」
    """

    # REPEAT-BREAKSHAPE
    payload, expected = _payload(
        count=6,
        repeats=4,
        sample_count=40,
        regime_count=4,
        breaks=(9, 19, 31),
        noise=0.002,
    )
    source = tmp_path / "four-regimes.json"
    output = tmp_path / "four-regimes-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["break_indices"] == [9, 19, 31]
    assert result["break_indices"] == expected["break_indices"]
    assert result["matches"] == expected["matches"]


def test_swapped_run_orders_do_not_follow_input_positions(tmp_path: Path) -> None:
    """题面：
    「`runs` 的长度必须等于 `repeats`。」
    """

    # REPEAT-RUNORDER REPEAT-MODEL
    payload, expected = _payload(breaks=(12, 27), noise=0.002)
    source = tmp_path / "run-order.json"
    output = tmp_path / "run-order-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["run_orders"] == expected["run_orders"]
    assert result["matches"] == expected["matches"]
    assert all(
        sorted(order) == [0, 1, 2, 3]
        for order in result["run_orders"].values()
    )


def test_drift_steps_follow_local_run_order_and_regimes(tmp_path: Path) -> None:
    """题面：
    「和 `drift_steps` 不得遗漏设备。」
    """

    # REPEAT-DRIFTSHAPE REPEAT-DRIFT-LOCAL REPEAT-ROUND
    payload, expected = _payload(
        count=6,
        repeats=3,
        sample_count=38,
        breaks=(10, 26),
        noise=0.0,
        tolerance=0.0001,
    )
    source = tmp_path / "drifts.json"
    output = tmp_path / "drifts-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    first_actual = [
        value
        for run in result["drift_steps"]["U-B9R3"]
        for value in run
    ]
    first_expected = [
        value
        for run in expected["drift_steps"]["U-B9R3"]
        for value in run
    ]
    last_actual = [
        value
        for run in result["drift_steps"]["U-F1T9"]
        for value in run
    ]
    last_expected = [
        value
        for run in expected["drift_steps"]["U-F1T9"]
        for value in run
    ]
    assert completed.returncode == 0
    assert first_actual == pytest.approx(first_expected, abs=1e-6, rel=0)
    assert last_actual == pytest.approx(last_expected, abs=1e-6, rel=0)
    assert len(result["drift_steps"]) == 6
    assert all(len(run_drifts) == 3 for run_drifts in result["drift_steps"].values())
    assert all(
        len(regime_drifts) == 2
        for run_drifts in result["drift_steps"].values()
        for regime_drifts in run_drifts
    )
    assert result["residual_rms"] == round(result["residual_rms"], 6)


def test_record_order_shuffle_preserves_recovered_structure(tmp_path: Path) -> None:
    """题面：
    「标准错误写一行 `REFUSED: <原因>`，退出状态为 2。成功输出使用紧凑 JSON，按字典序」
    """

    # REPEAT-GLOBALFIT REPEAT-NOTESTS
    payload, expected = _payload(noise=0.003)
    shuffled = dict(payload)
    shuffled["references"] = list(reversed(payload["references"]))
    shuffled["field_units"] = list(reversed(payload["field_units"]))
    source = tmp_path / "shuffled.json"
    output = tmp_path / "shuffled-result.json"
    _write_json(source, shuffled)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["matches"] == expected["matches"]
    assert result["run_orders"] == expected["run_orders"]
    assert result["break_indices"] == expected["break_indices"]


def test_nine_units_and_four_repeats_are_accepted(tmp_path: Path) -> None:
    """题面：
    「`references` 和 `field_units` 各含 6 到 9 条记录，两边数量相同。每条记录恰好有」
    """

    # REPEAT-COUNT-MAX
    payload, expected = _payload(
        count=9,
        repeats=4,
        sample_count=32,
        breaks=(6, 20),
        noise=0.002,
        tolerance=0.1,
    )
    source = tmp_path / "upper-count.json"
    output = tmp_path / "upper-count-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert len(result["matches"]) == 9
    assert len(result["run_orders"]) == 9
    assert result["matches"] == expected["matches"]


def test_long_series_and_minimum_collection_size_are_accepted(
    tmp_path: Path,
) -> None:
    """题面：
    「档案不合规或 `residual_rms` 超过 `residual_tolerance` 时，不创建输出文件。此时只在」
    """

    # REPEAT-RUNLEN-MAX
    payload, expected = _payload(
        count=6,
        repeats=3,
        sample_count=48,
        breaks=(14, 33),
        noise=0.0,
        tolerance=0.0001,
    )
    source = tmp_path / "long-series.json"
    output = tmp_path / "long-series-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["matches"] == expected["matches"]
    assert result["break_indices"] == [14, 33]
    assert result["residual_rms"] <= 0.0001


def test_break_vector_respects_six_sample_segments(tmp_path: Path) -> None:
    """题面：
    「入口是 `/app/predict.py`，调用时依次传入输入档案和输出路径。分析只使用镜像内自带的」
    """

    # REPEAT-SEGMENT
    payload, expected = _payload(
        count=6,
        repeats=3,
        sample_count=32,
        breaks=(6, 26),
        noise=0.001,
        tolerance=0.05,
    )
    source = tmp_path / "segment-floor.json"
    output = tmp_path / "segment-floor-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    edges = [0, *result["break_indices"], 32]
    assert completed.returncode == 0
    assert result["break_indices"] == [6, 26]
    assert result["break_indices"] == expected["break_indices"]
    assert all(edges[index + 1] - edges[index] >= 6 for index in range(3))


def test_extra_top_level_key_rejects_archive(tmp_path: Path) -> None:
    """题面：
    「编号，值是按设备本地 `runs` 顺序排列的基准轮次零基下标，每个下标恰好出现一次。」
    """

    # REPEAT-TOPKEYS REPEAT-TOPKEYS-02
    payload, _ = _payload(count=6, repeats=3)
    payload["batch_note"] = "unexpected"
    source = tmp_path / "extra-top.json"
    output = tmp_path / "extra-top-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_unknown_schema_is_refused(tmp_path: Path) -> None:
    """题面：
    「`U-[A-Z0-9]{4}`，同类编号不得重复。」
    """

    # REPEAT-SCHEMA REPEAT-REFUSAL
    payload, _ = _payload(count=6, repeats=3)
    payload["schema"] = "repeat-calibration-v1"
    source = tmp_path / "old-schema.json"
    output = tmp_path / "old-schema-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert completed.stderr.count("\n") == 1
    assert not output.exists()


def test_boolean_repeat_count_is_refused(tmp_path: Path) -> None:
    """题面：
    「## 交付格式」
    """

    # REPEAT-REPEATS
    payload, _ = _payload(count=6, repeats=3)
    payload["repeats"] = True
    source = tmp_path / "boolean-repeats.json"
    output = tmp_path / "boolean-repeats-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_boolean_series_value_is_refused(tmp_path: Path) -> None:
    """题面：
    「排列键，并保留一个行尾换行；同一输入重复运行时字节必须一致。」
    """

    # REPEAT-BOOL
    payload, _ = _payload(count=6, repeats=3)
    payload["field_units"][2]["runs"][1][7] = True
    source = tmp_path / "boolean-value.json"
    output = tmp_path / "boolean-value-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_repeat_run_collection_size_is_enforced(tmp_path: Path) -> None:
    """题面：
    「设备和轮次对应只能从输入序列计算，不能读取测试目录，也不能保存固定映射。」
    """

    # REPEAT-RUNCOUNT
    payload, _ = _payload(count=6, repeats=3)
    payload["field_units"][0]["runs"] = payload["field_units"][0]["runs"][:-1]
    source = tmp_path / "case-13.json"
    output = tmp_path / "result-13.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_extra_record_key_is_refused(tmp_path: Path) -> None:
    """题面：
    「多轮回放，把整只箱子对回基准工位，并还原箱内每一轮记录分别对应哪一轮回放。」
    """

    # REPEAT-RECORDKEYS
    payload, _ = _payload(count=6, repeats=3)
    payload["field_units"][0]["station_hint"] = "R-A7K2"
    source = tmp_path / "extra-record.json"
    output = tmp_path / "extra-record-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_invalid_unit_identifier_is_refused(tmp_path: Path) -> None:
    """题面：
    「断点和各轮系数。单轮曲线的相似程度只描述局部形状，不能代替整组映射和公共断点的」
    """

    # REPEAT-IDS
    payload, _ = _payload(count=6, repeats=3)
    payload["field_units"][0]["id"] = "U-12"
    source = tmp_path / "short-id.json"
    output = tmp_path / "short-id-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_unequal_collection_sizes_are_refused(tmp_path: Path) -> None:
    """题面：
    「入汛前的巡测车里留下了九只便携箱。每只箱子装过一组平行探针，箱内记录顺序在倒箱时」
    """

    # REPEAT-COUNTS
    payload, _ = _payload(count=9, repeats=4, sample_count=32, breaks=(6, 20))
    payload["field_units"] = payload["field_units"][:-1]
    source = tmp_path / "unequal-count.json"
    output = tmp_path / "unequal-count-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_short_run_is_refused(tmp_path: Path) -> None:
    """题面：
    「全部设备编号，值按设备本地轮次排列；每个元素再按公共断点顺序给出各段新增的阶跃量。」
    """

    # REPEAT-RUNLEN
    payload, _ = _payload(count=6, repeats=3, sample_count=32, breaks=(6, 20))
    payload["field_units"][0]["runs"][0] = payload["field_units"][0]["runs"][0][:-1]
    source = tmp_path / "short-run.json"
    output = tmp_path / "short-run-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_unequal_series_lengths_are_refused(tmp_path: Path) -> None:
    """题面：
    「`residual_tolerance` 六项。`schema` 必须是 `"repeat-calibration-v2"`。」
    """

    # REPEAT-EQUAL
    payload, _ = _payload(count=6, repeats=3, sample_count=32, breaks=(6, 20))
    payload["field_units"][0]["runs"][0].append(0.0)
    source = tmp_path / "unequal-length.json"
    output = tmp_path / "unequal-length-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_out_of_range_measurement_is_refused(tmp_path: Path) -> None:
    """题面：
    「`residual_tolerance` 六项。`schema` 必须是 `"repeat-calibration-v2"`。」
    """

    # REPEAT-VALUE
    payload, _ = _payload(count=6, repeats=3, sample_count=32, breaks=(6, 20))
    payload["field_units"][0]["runs"][0][0] = 100.5
    source = tmp_path / "range-value.json"
    output = tmp_path / "range-value-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_residual_tolerance_failure_is_refused(tmp_path: Path) -> None:
    """题面：
    「`residual_tolerance` 六项。`schema` 必须是 `"repeat-calibration-v2"`。」
    """

    # REPEAT-TOLERANCE
    payload, _ = _payload(
        count=6,
        repeats=3,
        sample_count=32,
        breaks=(6, 20),
        noise=0.35,
        tolerance=0.0001,
    )
    source = tmp_path / "high-residual.json"
    output = tmp_path / "high-residual-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_output_has_exact_fields_and_declared_rms(tmp_path: Path) -> None:
    """题面：
    「`references` 和 `field_units` 各含 6 到 9 条记录，两边数量相同。每条记录恰好有」
    """

    # REPEAT-NO-MISSING REPEAT-ONETOONE REPEAT-RMS REPEAT-OUTKEYS
    # REPEAT-MATCHKEYS
    payload, _ = _payload(noise=0.0, tolerance=0.05)
    source = tmp_path / "fields.json"
    output = tmp_path / "fields-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert set(result) == {
        "break_indices",
        "drift_steps",
        "matches",
        "residual_rms",
        "run_orders",
    }
    assert set(result["matches"]) == set(FIELD_IDS[:8])
    assert set(result["matches"].values()) == set(REFERENCE_IDS[:8])
    assert all(sorted(order) == [0, 1, 2, 3] for order in result["run_orders"].values())
    assert result["residual_rms"] <= 0.0001


def test_output_is_compact_sorted_and_repeatable(tmp_path: Path) -> None:
    """题面：
    「`residual_tolerance` 六项。`schema` 必须是 `"repeat-calibration-v2"`。」
    """

    # REPEAT-CLI REPEAT-FORMAT REPEAT-REPEATABLE REPEAT-STDLIB
    payload, _ = _payload(
        count=6,
        repeats=3,
        sample_count=32,
        breaks=(6, 20),
        noise=0.002,
    )
    source = tmp_path / "format.json"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_json(source, payload)
    first_run = _run(source, first)
    second_run = _run(source, second)
    raw = first.read_bytes()
    parsed = json.loads(raw)
    canonical = (
        json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    assert first_run.returncode == 0
    assert second_run.returncode == 0
    assert raw == canonical
    assert raw == second.read_bytes()
