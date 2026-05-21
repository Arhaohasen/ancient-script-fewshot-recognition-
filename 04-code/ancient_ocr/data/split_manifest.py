from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _with_label_index(records: list[dict[str, Any]], label_to_index: dict[str, int]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["label_index"] = label_to_index[item["label"]]
        output.append(item)
    return output


def split_manifest(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    valid_fraction: float = 0.1,
    seed: int = 42,
    min_train_per_label: int = 1,
) -> dict[str, Any]:
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1")
    if min_train_per_label < 1:
        raise ValueError("min_train_per_label must be at least 1")

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(manifest_path)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)

    rng = random.Random(seed)
    train_records: list[dict[str, Any]] = []
    valid_records: list[dict[str, Any]] = []

    for label in sorted(by_label):
        label_records = sorted(by_label[label], key=lambda item: item["crop_path"])
        rng.shuffle(label_records)
        max_valid = max(0, len(label_records) - min_train_per_label)
        if max_valid == 0:
            valid_count = 0
        else:
            valid_count = min(max_valid, max(1, round(len(label_records) * valid_fraction)))
        valid_records.extend(label_records[:valid_count])
        train_records.extend(label_records[valid_count:])

    train_records = sorted(train_records, key=lambda item: (item["label"], item["crop_path"]))
    valid_records = sorted(valid_records, key=lambda item: (item["label"], item["crop_path"]))
    labels = sorted({record["label"] for record in train_records})
    label_to_index = {label: index for index, label in enumerate(labels)}

    train_records = _with_label_index(train_records, label_to_index)
    valid_records = _with_label_index(valid_records, label_to_index)

    summary: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "seed": seed,
        "valid_fraction": valid_fraction,
        "min_train_per_label": min_train_per_label,
        "total_records": len(records),
        "train_records": len(train_records),
        "valid_records": len(valid_records),
        "num_classes": len(labels),
        "train_labels": len({record["label"] for record in train_records}),
        "valid_labels": len({record["label"] for record in valid_records}),
        "singleton_labels": sum(1 for label_records in by_label.values() if len(label_records) == 1),
    }

    _write_jsonl(output_dir / "train.jsonl", train_records)
    _write_jsonl(output_dir / "valid.jsonl", valid_records)
    (output_dir / "label_to_index.json").write_text(
        json.dumps(label_to_index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic train/valid splits from crop manifest.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory for split files.")
    parser.add_argument("--valid-fraction", type=float, default=0.1, help="Per-label validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--min-train-per-label", type=int, default=1, help="Minimum training records per label.")
    args = parser.parse_args()

    summary = split_manifest(
        args.manifest,
        output_dir=args.output_dir,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
        min_train_per_label=args.min_train_per_label,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
