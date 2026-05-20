import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from ancient_ocr.data.build_crops import (  # noqa: E402
    extract_crops,
    find_image_for_xml,
    label_codepoints,
)


class BuildCropsTests(unittest.TestCase):
    def test_label_codepoints_uses_unicode_code_points(self):
        self.assertEqual(label_codepoints("大"), "U+5927")
        self.assertEqual(label_codepoints("𢭃"), "U+22B43")

    def test_finds_image_with_supported_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "sample.xml"
            png_path = Path(tmp) / "sample.png"
            xml_path.write_text("<page />")
            png_path.write_bytes(b"not a real image")

            self.assertEqual(find_image_for_xml(xml_path), png_path)

    def test_extracts_valid_char_crop_and_manifest(self):
        xml = """<?xml version="1.0" encoding="utf-16" standalone="yes"?>
<page id="sample.png" width="100" height="100">
  <text><text_row no="1"><chunk><char id="7" position="10,20;30,20;30,50;10,50">大</char></chunk></text_row></text>
</page>
"""
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "train"
            output_dir = Path(tmp) / "out"
            data_root.mkdir()
            (data_root / "sample.xml").write_bytes(xml.encode("utf-16"))
            Image.new("RGB", (100, 100), "white").save(data_root / "sample.png")

            summary = extract_crops(
                data_root=data_root,
                output_dir=output_dir,
                limit=1,
                padding=0,
            )

            manifest_path = output_dir / "manifest.jsonl"
            records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
            crop_exists = (output_dir / records[0]["crop_path"]).exists()

        self.assertEqual(summary["crops_written"], 1)
        self.assertEqual(records[0]["label"], "大")
        self.assertEqual(records[0]["label_codepoints"], "U+5927")
        self.assertEqual(records[0]["bbox"], [10, 20, 30, 50])
        self.assertTrue(crop_exists)


if __name__ == "__main__":
    unittest.main()
