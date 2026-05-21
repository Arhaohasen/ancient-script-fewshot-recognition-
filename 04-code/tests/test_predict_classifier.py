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
    build_arg_parser,
    build_prediction_payload,
    detect_candidate_boxes,
    find_input_images,
    resolve_image_dir,
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

    def test_resolves_sais_eval_image_dir_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "13" / "eval" / "images"
            eval_dir.mkdir(parents=True)

            self.assertEqual(resolve_image_dir(root), eval_dir)

    def test_input_image_id_uses_png_stem_for_sais_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "13" / "eval" / "images"
            eval_dir.mkdir(parents=True)
            Image.new("RGB", (8, 8), "white").save(eval_dir / "IMG001.png")

            images = find_input_images(root)

        self.assertEqual(images[0].image_id, "IMG001")
        self.assertEqual(images[0].relative_path, "IMG001.png")

    def test_builds_sais_detection_payload(self):
        payload = build_prediction_payload(
            predictions=[
                (
                    "IMG001",
                    [
                        {"bbox": [1, 2, 3, 4], "text": "甲"},
                        {"bbox": [5, 6, 7, 8], "text": "乙"},
                    ],
                )
            ],
            schema="sais",
        )

        self.assertEqual(
            payload,
            {
                "IMG001": [
                    {"bbox": [1, 2, 3, 4], "text": "甲"},
                    {"bbox": [5, 6, 7, 8], "text": "乙"},
                ]
            },
        )

    def test_detect_candidate_boxes_finds_simple_dark_component(self):
        image = Image.new("L", (40, 30), "white")
        pixels = image.load()
        for y in range(8, 18):
            for x in range(10, 22):
                pixels[x, y] = 0

        boxes = detect_candidate_boxes(image, min_area=20, max_boxes=10)

        self.assertEqual(len(boxes), 1)
        self.assertLessEqual(boxes[0][0], 10)
        self.assertLessEqual(boxes[0][1], 8)
        self.assertGreaterEqual(boxes[0][2], 12)
        self.assertGreaterEqual(boxes[0][3], 10)

    def test_detect_candidate_boxes_filters_page_sized_components(self):
        image = Image.new("L", (100, 100), "white")
        pixels = image.load()
        for y in range(5, 95):
            pixels[5, y] = 0
        for y in range(40, 50):
            for x in range(60, 70):
                pixels[x, y] = 0

        boxes = detect_candidate_boxes(image, min_area=20, max_boxes=10)

        self.assertEqual(len(boxes), 1)
        self.assertGreaterEqual(boxes[0][0], 55)
        self.assertLessEqual(boxes[0][0], 60)

    def test_writes_utf8_prediction_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prediction.json"

            write_prediction_json(path, {"IMG001": [{"bbox": [1, 2, 3, 4], "text": "𢭃"}]})

            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded, {"IMG001": [{"bbox": [1, 2, 3, 4], "text": "𢭃"}]})

    def test_arg_parser_exposes_fast_inference_controls(self):
        parser = build_arg_parser()

        args = parser.parse_args(
            [
                "--checkpoint",
                "/app/best.pt",
                "--max-boxes",
                "64",
                "--max-detection-side",
                "800",
                "--progress-interval",
                "25",
            ]
        )

        self.assertEqual(args.max_boxes, 64)
        self.assertEqual(args.max_detection_side, 800)
        self.assertEqual(args.progress_interval, 25)


if __name__ == "__main__":
    unittest.main()
