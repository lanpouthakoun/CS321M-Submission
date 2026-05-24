
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from torch_measure.datasets import LongFormData, list_datasets, load


def parse_subject_name(subject_content: str) -> str:
    if not subject_content:
        return ""
    for line in subject_content.splitlines():
        if line.startswith("Name:"):
            return line[len("Name:"):].strip()
    for line in subject_content.splitlines():
        if line.strip():
            return line.strip()
    return ""


def render_subject_content(subject: dict, fallback_subject_id: str) -> str:
    display_name = subject.get("display_name") or fallback_subject_id
    lines = [f"Name: {display_name}"]
    optional_fields = (
        ("provider", "Organization"),
        ("params", "Parameters"),
        ("release_date", "Released"),
        ("family", "Family"),
    )
    for key, label in optional_fields:
        value = subject.get(key)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def benchmark_to_long_frame(data: LongFormData) -> pd.DataFrame:
    items_by_id = {row["item_id"]: row for row in data.items.to_dict("records")}
    subjects_by_id = {row["subject_id"]: row for row in data.subjects.to_dict("records")}

    out = data.responses[
        ["subject_id", "item_id", "benchmark_id", "test_condition", "response"]
    ].copy()
    out["benchmark"] = out["benchmark_id"]

    subject_content = out["subject_id"].map(
        lambda sid: render_subject_content(subjects_by_id.get(sid, {}), sid)
    )
    out["subject_content"] = subject_content
    out["subject_name"] = subject_content.map(parse_subject_name)
    out["item_content"] = out["item_id"].map(
        lambda iid: items_by_id.get(iid, {}).get("content")
    )
    out["condition"] = out["test_condition"].fillna("none").replace("", "none")
    out["label"] = out["response"]
    return out[
        [
            "benchmark", "condition",
            "subject_id", "subject_name",
            "item_id",
            "subject_content", "item_content",
            "label",
        ]
    ]


def build_training_table(out_path: Path, dataset_names: list[str] | None = None) -> pd.DataFrame:
    names = dataset_names or list_datasets()
    print(f"Loading {len(names)} datasets via torch_measure.datasets.load(): {names}")

    frames: list[pd.DataFrame] = []
    for name in names:
        data = load(name)
        n_resp = len(data.responses)
        if n_resp == 0:
            print(f"  - {name}: empty, skipping")
            continue
        frames.append(benchmark_to_long_frame(data))
        print(f"  - {name}: {n_resp:,} responses, "
              f"{data.responses['subject_id'].nunique():,} subjects, "
              f"{data.responses['item_id'].nunique():,} items")

    df = pd.concat(frames, ignore_index=True)
    print(
        f"Combined long-form table: {len(df):,} rows, "
        f"{df['subject_id'].nunique():,} subjects, "
        f"{df['item_id'].nunique():,} items, "
        f"{df['benchmark'].nunique():,} benchmarks"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "training_long.parquet",
        help="Where to write the joined long-form parquet.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional subset of dataset names. Default: all from list_datasets().",
    )
    args = parser.parse_args()
    build_training_table(args.out, args.datasets)


if __name__ == "__main__":
    main()
