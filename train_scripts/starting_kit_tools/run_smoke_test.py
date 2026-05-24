#!/usr/bin/env python3
"""Run a small local smoke test against an unpacked submission directory."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import numbers
import os
import sys
from pathlib import Path


KIT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_TEST_DIR = KIT_DIR / "sample_data" / "test"
SAMPLE_REF_DIR = KIT_DIR / "sample_data" / "ref"
LOCAL_SMOKE_TEST_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_dir", type=Path)
    args = parser.parse_args()

    try:
        result = run_smoke_test(args.submission_dir)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "OK: smoke test passed "
        f"({int(result['prediction_count'])} predictions, log_loss={result['log_loss']:.6f})."
    )
    return 0


def run_smoke_test(submission_dir: Path) -> dict[str, float]:
    submission_dir = submission_dir.resolve()
    model_path = submission_dir / "model.py"
    if not model_path.exists():
        raise ValueError(f"model.py not found at {model_path}")

    test_rows = _load_csv(SAMPLE_TEST_DIR / "test_items.csv", key="item_id")
    model_rows = _load_csv(SAMPLE_TEST_DIR / "models.csv", key="model_id")
    bench_conditions = {
        row["benchmark_name"]: row.get("test_condition", "")
        for row in _load_csv_rows(SAMPLE_TEST_DIR / "benchmarks.csv")
    }
    pairs = _load_csv_rows(SAMPLE_TEST_DIR / "test_pairs.csv")
    labels = {
        (row["model_id"], row["item_id"]): int(row["label"])
        for row in _load_csv_rows(SAMPLE_REF_DIR / "track1_ground_truth.csv")
    }

    inputs = [
        _build_input(pair, test_rows, model_rows, bench_conditions)
        for pair in pairs
    ]

    labeling_path = submission_dir / "labeling.py"
    labeled = []
    if labeling_path.exists():
        labeling = _load_module(labeling_path, "smoke_labeling", submission_dir)
        acquisition = getattr(labeling, "acquisition_function", None)
        if not callable(acquisition):
            raise ValueError("labeling.py must define callable acquisition_function(input).")
        for input_row in inputs:
            _assert_finite_number(acquisition(dict(input_row)), "acquisition_function()")
    if inputs:
        labeled = [dict(inputs[0], label=labels[(pairs[0]["model_id"], pairs[0]["item_id"])])]

    model = _load_module(model_path, "smoke_model", submission_dir)
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise ValueError("model.py must define callable predict(input, labeled=None).")

    predictions: list[float] = []
    truth: list[int] = []
    for pair, input_row in zip(pairs, inputs):
        try:
            value = predict(dict(input_row), labeled=list(labeled))
        except TypeError as exc:
            raise ValueError("predict() must accept the labeled keyword argument.") from exc
        score = _assert_probability(value, "predict()")
        predictions.append(score)
        truth.append(labels[(pair["model_id"], pair["item_id"])])

    return {
        "prediction_count": len(predictions),
        "log_loss": _log_loss(truth, predictions),
    }


def _load_csv(path: Path, *, key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in _load_csv_rows(path) if row.get(key)}


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"Sample data file missing: {path}")
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _build_input(
    pair: dict[str, str],
    item_rows: dict[str, dict[str, str]],
    model_rows: dict[str, dict[str, str]],
    bench_conditions: dict[str, str],
) -> dict[str, str]:
    item = item_rows.get(pair["item_id"], {})
    model = model_rows.get(pair["model_id"], {})
    benchmark = item.get("benchmark", "")
    return {
        "benchmark": benchmark,
        "condition": item.get("condition") or bench_conditions.get(benchmark) or "none",
        "subject_content": _render_subject(pair["model_id"], model),
        "item_content": item.get("item_text", ""),
    }


def _render_subject(model_id: str, model: dict[str, str]) -> str:
    lines = [f"Name: {model.get('name') or model_id}"]
    for key, label in [
        ("organization", "Organization"),
        ("size_params", "Parameters"),
        ("release_date", "Released"),
        ("family", "Family"),
    ]:
        value = model.get(key)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _load_module(path: Path, module_name: str, submission_dir: Path):
    previous_path = list(sys.path)
    previous_env = os.environ.get(LOCAL_SMOKE_TEST_ENV)
    sys.path.insert(0, str(submission_dir))
    os.environ[LOCAL_SMOKE_TEST_ENV] = "1"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not import {path.name}.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_env is None:
            os.environ.pop(LOCAL_SMOKE_TEST_ENV, None)
        else:
            os.environ[LOCAL_SMOKE_TEST_ENV] = previous_env
        sys.path[:] = previous_path


def _assert_probability(value, label: str) -> float:
    number = _assert_finite_number(value, label)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{label} must return a probability in [0, 1].")
    return number


def _assert_finite_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must return a finite numeric value.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must return a finite numeric value.")
    return number


def _log_loss(labels: list[int], predictions: list[float]) -> float:
    eps = 1e-7
    total = 0.0
    for label, prediction in zip(labels, predictions):
        p = min(max(prediction, eps), 1.0 - eps)
        total += -(label * math.log(p) + (1 - label) * math.log(1 - p))
    return total / max(len(labels), 1)


if __name__ == "__main__":
    raise SystemExit(main())
