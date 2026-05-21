import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from ancient_ocr.training.train_classifier import (  # noqa: E402
    build_run_config,
    load_index_to_label,
    read_jsonl,
)


class TrainClassifierTests(unittest.TestCase):
    def test_read_jsonl_supports_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text(
                "\n".join(json.dumps({"i": index}) for index in range(3)) + "\n",
                encoding="utf-8",
            )

            records = read_jsonl(path, limit=2)

        self.assertEqual(records, [{"i": 0}, {"i": 1}])

    def test_load_index_to_label_uses_label_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "label_to_index.json"
            path.write_text(json.dumps({"乙": 1, "甲": 0}, ensure_ascii=False), encoding="utf-8")

            labels = load_index_to_label(path)

        self.assertEqual(labels, ["甲", "乙"])

    def test_build_run_config_records_reproducible_settings(self):
        config = build_run_config(
            data_root="/data/crops",
            train_split="/data/splits/train.jsonl",
            valid_split="/data/splits/valid.jsonl",
            label_map="/data/splits/label_to_index.json",
            output_dir="/runs/exp002",
            model="resnet50",
            weights="none",
            image_size=224,
            batch_size=32,
            epochs=1,
            lr=0.001,
            seed=42,
            num_workers=4,
            amp=True,
            limit_train=100,
            limit_valid=50,
        )

        self.assertEqual(config["model"], "resnet50")
        self.assertEqual(config["weights"], "none")
        self.assertEqual(config["image_size"], 224)
        self.assertEqual(config["limit_train"], 100)
        self.assertTrue(config["amp"])


if __name__ == "__main__":
    unittest.main()
