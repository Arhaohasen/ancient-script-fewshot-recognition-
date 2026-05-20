import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from ancient_ocr.data.summarize_manifest import summarize_manifest  # noqa: E402


class SummarizeManifestTests(unittest.TestCase):
    def test_summarizes_label_counts_and_long_tail(self):
        records = [
            {"label": "大", "label_codepoints": "U+5927"},
            {"label": "大", "label_codepoints": "U+5927"},
            {"label": "小", "label_codepoints": "U+5C0F"},
            {"label": "𢭃", "label_codepoints": "U+22B43"},
            {"label": "AB", "label_codepoints": "U+0041 U+0042"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
                encoding="utf-8",
            )

            summary = summarize_manifest(manifest_path)

        self.assertEqual(summary["total_crops"], 5)
        self.assertEqual(summary["unique_labels"], 4)
        self.assertEqual(summary["singleton_labels"], 3)
        self.assertEqual(summary["multi_codepoint_labels"], 1)
        self.assertEqual(summary["top_labels"][0]["label"], "大")
        self.assertEqual(summary["top_labels"][0]["count"], 2)

    def test_writes_summary_and_label_counts_when_output_dir_is_given(self):
        records = [
            {"label": "甲", "label_codepoints": "U+7532"},
            {"label": "乙", "label_codepoints": "U+4E59"},
            {"label": "乙", "label_codepoints": "U+4E59"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.jsonl"
            output_dir = root / "stats"
            manifest_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
                encoding="utf-8",
            )

            summarize_manifest(manifest_path, output_dir=output_dir)

            summary = json.loads((output_dir / "dataset_summary.json").read_text(encoding="utf-8"))
            label_counts = [
                json.loads(line)
                for line in (output_dir / "label_counts.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(summary["total_crops"], 3)
        self.assertEqual(label_counts[0]["label"], "乙")
        self.assertEqual(label_counts[0]["count"], 2)


if __name__ == "__main__":
    unittest.main()
