import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from ancient_ocr.inference.predict_classifier import (  # noqa: E402
    build_prediction_payload,
    find_input_images,
    write_prediction_json,
)


class PredictClassifierTests(unittest.TestCase):
    def test_find_input_images_returns_sorted_supported_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            Image.new("RGB", (8, 8), "white").save(root / "b.png")
            Image.new("RGB", (8, 8), "white").save(root / "nested" / "a.jpg")
            (root / "ignore.txt").write_text("nope", encoding="utf-8")

            images = find_input_images(root)

        self.assertEqual([item.relative_path for item in images], ["b.png", "nested/a.jpg"])

    def test_builds_mapping_payload(self):
        payload = build_prediction_payload(
            predictions=[("a.png", "甲"), ("b.png", "乙")],
            schema="mapping",
        )

        self.assertEqual(payload, {"a.png": "甲", "b.png": "乙"})

    def test_builds_records_payload(self):
        payload = build_prediction_payload(
            predictions=[("a.png", "甲")],
            schema="records",
        )

        self.assertEqual(payload, [{"image_id": "a.png", "prediction": "甲"}])

    def test_writes_utf8_prediction_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prediction.json"

            write_prediction_json(path, {"a.png": "𢭃"})

            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded, {"a.png": "𢭃"})


if __name__ == "__main__":
    unittest.main()
