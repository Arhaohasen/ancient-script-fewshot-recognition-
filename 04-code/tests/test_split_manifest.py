import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from ancient_ocr.data.split_manifest import split_manifest  # noqa: E402


def _record(label: str, index: int) -> dict[str, object]:
    return {
        "crop_path": f"crops/{label}_{index}.png",
        "label": label,
        "label_codepoints": " ".join(f"U+{ord(char):04X}" for char in label),
    }


class SplitManifestTests(unittest.TestCase):
    def test_keeps_singleton_labels_in_train_only(self):
        records = [
            _record("single", 1),
            _record("pair", 1),
            _record("pair", 2),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.jsonl"
            output_dir = root / "splits"
            manifest_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
                encoding="utf-8",
            )

            summary = split_manifest(
                manifest_path,
                output_dir=output_dir,
                valid_fraction=0.5,
                seed=7,
            )

            train_records = [
                json.loads(line)
                for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            valid_records = [
                json.loads(line)
                for line in (output_dir / "valid.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(summary["train_records"], 2)
        self.assertEqual(summary["valid_records"], 1)
        self.assertEqual([record["label"] for record in valid_records], ["pair"])
        self.assertIn("single", {record["label"] for record in train_records})

    def test_filters_placeholder_labels_from_existing_manifest(self):
        records = [
            _record("甲", 1),
            _record("甲", 2),
            _record("ZHFD-1176-1528961428922", 1),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.jsonl"
            output_dir = root / "splits"
            manifest_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
                encoding="utf-8",
            )

            summary = split_manifest(manifest_path, output_dir=output_dir, valid_fraction=0.5, seed=7)

            label_map = json.loads((output_dir / "label_to_index.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(label_map, {"甲": 0})

    def test_writes_label_map_from_all_train_labels(self):
        records = [
            _record("乙", 1),
            _record("甲", 1),
            _record("甲", 2),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.jsonl"
            output_dir = root / "splits"
            manifest_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
                encoding="utf-8",
            )

            split_manifest(manifest_path, output_dir=output_dir, valid_fraction=0.5, seed=1)

            label_map = json.loads((output_dir / "label_to_index.json").read_text(encoding="utf-8"))
            summary = json.loads((output_dir / "split_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(label_map, {"乙": 0, "甲": 1})
        self.assertEqual(summary["num_classes"], 2)
        self.assertEqual(summary["valid_labels"], 1)

    def test_split_is_deterministic_for_same_seed(self):
        records = [_record("many", index) for index in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.jsonl"
            first_dir = root / "first"
            second_dir = root / "second"
            manifest_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
                encoding="utf-8",
            )

            split_manifest(manifest_path, output_dir=first_dir, valid_fraction=0.2, seed=42)
            split_manifest(manifest_path, output_dir=second_dir, valid_fraction=0.2, seed=42)

            first_valid = (first_dir / "valid.jsonl").read_text(encoding="utf-8")
            second_valid = (second_dir / "valid.jsonl").read_text(encoding="utf-8")

        self.assertEqual(first_valid, second_valid)


if __name__ == "__main__":
    unittest.main()
