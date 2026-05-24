#!/usr/bin/env python3
"""Validate a Predictive Evaluation Challenge submission ZIP."""

from __future__ import annotations

import argparse
import importlib.util
import math
import numbers
import os
import sys
import tempfile
import zipfile
from pathlib import Path


MAX_MODELS = 5
LOCAL_SMOKE_TEST_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"
SMOKE_INPUT = {
    "benchmark": "sample_benchmark",
    "condition": "none",
    "subject_content": "Name: sample-model\nOrganization: sample-org",
    "item_content": "A sample yes/no evaluation item.",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_zip", type=Path)
    args = parser.parse_args()

    try:
        validate_submission_zip(args.submission_zip)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"OK: {args.submission_zip} looks like a valid submission ZIP.")
    return 0


def validate_submission_zip(zip_path: Path) -> None:
    if not zip_path.exists():
        raise ValueError(f"ZIP not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid ZIP file: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        normalized = {name.replace("\\", "/") for name in names}
        _reject_unsafe_members(normalized)

        if any(name.lower().endswith(".zip") for name in normalized):
            raise ValueError("Do not upload a ZIP that contains another ZIP. Upload the submission files directly.")

        if "model.py" not in normalized:
            nested_model = sorted(name for name in normalized if name.endswith("/model.py"))
            if nested_model:
                raise ValueError(
                    "model.py is nested inside a folder. Zip the contents of your submission directory, "
                    "not the directory itself."
                )
            raise ValueError("model.py must be at the ZIP root.")

        if "models.txt" not in normalized and any(name.endswith("/models.txt") for name in normalized):
            raise ValueError("models.txt is nested inside a folder. It must be at the ZIP root.")

        models = _read_models_txt(zf, normalized)
        if len(models) > MAX_MODELS:
            raise ValueError(f"models.txt lists {len(models)} models; maximum allowed is {MAX_MODELS}.")

        with tempfile.TemporaryDirectory(prefix="submission-check-") as tmpdir:
            zf.extractall(tmpdir)
            submission_dir = Path(tmpdir)
            _check_model(submission_dir)
            _check_labeling(submission_dir)


def _reject_unsafe_members(names: set[str]) -> None:
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("ZIP contains an unsafe file path. Recreate it from the submission directory contents.")


def _read_models_txt(zf: zipfile.ZipFile, names: set[str]) -> list[str]:
    if "models.txt" not in names:
        return []
    with zf.open("models.txt") as handle:
        text = handle.read().decode("utf-8", errors="replace")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _check_model(submission_dir: Path) -> None:
    model = _load_module(submission_dir / "model.py", "submission_model", submission_dir)
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise ValueError("model.py must define callable predict(input, labeled=None).")
    try:
        value = predict(dict(SMOKE_INPUT), labeled=[])
    except TypeError as exc:
        raise ValueError("predict() must accept the labeled keyword argument.") from exc
    except Exception as exc:
        raise ValueError("predict() raised during the local smoke check.") from exc
    _assert_finite_probability(value, "predict()")


def _check_labeling(submission_dir: Path) -> None:
    labeling_path = submission_dir / "labeling.py"
    if not labeling_path.exists():
        return
    labeling = _load_module(labeling_path, "submission_labeling", submission_dir)
    acquisition = getattr(labeling, "acquisition_function", None)
    if not callable(acquisition):
        raise ValueError("labeling.py must define callable acquisition_function(input).")
    try:
        value = acquisition(dict(SMOKE_INPUT))
    except Exception as exc:
        raise ValueError("acquisition_function() raised during the local smoke check.") from exc
    _assert_finite_number(value, "acquisition_function()")


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


def _assert_finite_probability(value, label: str) -> None:
    number = _assert_finite_number(value, label)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{label} must return a probability in [0, 1].")


def _assert_finite_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must return a finite numeric value.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must return a finite numeric value.")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
