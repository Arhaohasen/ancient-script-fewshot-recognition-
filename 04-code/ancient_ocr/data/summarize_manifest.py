from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _label_counts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(record["label"] for record in records)
    codepoints = {
        record["label"]: record.get("label_codepoints", "")
        for record in records
    }
    return [
        {
            "label": label,
            "label_codepoints": codepoints[label],
            "count": count,
        }
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_manifest(
    manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    records = _read_manifest(manifest_path)
    label_counts = _label_counts(records)
    counts = [record["count"] for record in label_counts]

    summary: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "total_crops": len(records),
        "unique_labels": len(label_counts),
        "singleton_labels": sum(1 for count in counts if count == 1),
        "multi_codepoint_labels": sum(1 for record in label_counts if len(record["label"]) > 1),
        "min_count": min(counts) if counts else 0,
        "max_count": max(counts) if counts else 0,
        "top_labels": label_counts[:top_k],
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_jsonl(output_dir / "label_counts.jsonl", label_counts)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize crop manifest label distribution.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl.")
    parser.add_argument("--output-dir", default=None, help="Optional directory for summary files.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top labels in summary JSON.")
    args = parser.parse_args()

    summary = summarize_manifest(
        args.manifest,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
