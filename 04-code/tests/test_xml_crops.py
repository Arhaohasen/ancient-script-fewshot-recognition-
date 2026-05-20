import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from ancient_ocr.data.xml_crops import (  # noqa: E402
    bbox_from_points,
    infer_coordinate_transform,
    is_valid_label,
    load_xml_page,
    map_xml_points_to_image,
    parse_position,
)


class XmlCropTests(unittest.TestCase):
    def test_filters_invalid_labels(self):
        self.assertFalse(is_valid_label(None))
        self.assertFalse(is_valid_label(""))
        self.assertFalse(is_valid_label("   "))
        self.assertFalse(is_valid_label("None"))
        self.assertFalse(is_valid_label("ZH-12345"))
        self.assertTrue(is_valid_label("大"))

    def test_parses_semicolon_position_points(self):
        self.assertEqual(
            parse_position("10,20;30,40;50,60"),
            [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)],
        )

    def test_scales_xml_bbox_to_image_coordinates(self):
        points = [(10.0, 20.0), (30.0, 40.0)]
        bbox = bbox_from_points(
            points,
            xml_size=(100, 100),
            image_size=(200, 300),
            padding=5,
        )
        self.assertEqual(bbox, (15, 55, 65, 125))

    def test_infers_direct_transform_when_scales_match(self):
        self.assertEqual(
            infer_coordinate_transform(xml_size=(100, 200), image_size=(300, 600)),
            "direct",
        )

    def test_infers_clockwise_rotation_when_dimensions_are_swapped(self):
        self.assertEqual(
            infer_coordinate_transform(xml_size=(2733, 1232), image_size=(1242, 2733)),
            "rotate_cw",
        )

    def test_maps_clockwise_rotated_xml_points_to_image_coordinates(self):
        mapped = map_xml_points_to_image(
            [(2032.0, 384.0), (2089.0, 457.0)],
            xml_size=(2733, 1232),
            image_size=(1242, 2733),
            coordinate_transform="rotate_cw",
        )
        self.assertAlmostEqual(mapped[0][0], 854.9, places=1)
        self.assertAlmostEqual(mapped[0][1], 2032.0, places=1)
        self.assertAlmostEqual(mapped[1][0], 781.3, places=1)
        self.assertAlmostEqual(mapped[1][1], 2089.0, places=1)

    def test_loads_utf16_xml_using_declaration(self):
        xml = """<?xml version="1.0" encoding="utf-16" standalone="yes"?>
<page id="sample.png" width="100" height="80">
  <text><text_row no="1"><chunk><char id="1" position="1,2;3,4">大</char></chunk></text_row></text>
</page>
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.xml"
            path.write_bytes(xml.encode("utf-16"))

            page = load_xml_page(path)

        self.assertEqual(page.image_id, "sample.png")
        self.assertEqual(page.xml_size, (100, 80))
        self.assertEqual(len(page.chars), 1)
        self.assertEqual(page.chars[0].label, "大")
        self.assertEqual(page.chars[0].points, [(1.0, 2.0), (3.0, 4.0)])


if __name__ == "__main__":
    unittest.main()
