from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class CharBox:
    label: str | None
    points: list[tuple[float, float]]
    attrs: dict[str, str]


@dataclass(frozen=True)
class XmlPage:
    path: Path
    image_id: str | None
    xml_size: tuple[int, int]
    chars: list[CharBox]


def is_valid_label(label: str | None) -> bool:
    if label is None:
        return False
    value = label.strip()
    if not value:
        return False
    if value == "None":
        return False
    if value.startswith("ZH-"):
        return False
    if re.match(r"^[A-Za-z]+[-_]", value):
        return False
    return True


def parse_position(position: str) -> list[tuple[float, float]]:
    if ";" not in position:
        values = [float(value.strip()) for value in position.split(",") if value.strip()]
        if len(values) == 4:
            left, top, right, bottom = values
            return [(left, top), (right, top), (right, bottom), (left, bottom)]

    points: list[tuple[float, float]] = []
    for pair in position.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x_text, y_text = pair.split(",", 1)
        points.append((float(x_text), float(y_text)))
    return points


def _scale_score(scale_a: float, scale_b: float) -> float:
    if scale_a <= 0 or scale_b <= 0:
        return math.inf
    return abs(math.log(scale_a / scale_b))


def infer_coordinate_transform(
    *,
    xml_size: tuple[int, int],
    image_size: tuple[int, int],
) -> str:
    xml_width, xml_height = xml_size
    image_width, image_height = image_size
    if min(xml_width, xml_height, image_width, image_height) <= 0:
        raise ValueError(f"invalid sizes: xml={xml_size}, image={image_size}")

    direct_score = _scale_score(image_width / xml_width, image_height / xml_height)
    rotate_score = _scale_score(image_width / xml_height, image_height / xml_width)
    if direct_score <= rotate_score:
        return "direct"
    return "rotate_cw"


def map_xml_points_to_image(
    points: list[tuple[float, float]],
    *,
    xml_size: tuple[int, int],
    image_size: tuple[int, int],
    coordinate_transform: str = "auto",
) -> list[tuple[float, float]]:
    xml_width, xml_height = xml_size
    image_width, image_height = image_size
    if coordinate_transform == "auto":
        coordinate_transform = infer_coordinate_transform(
            xml_size=xml_size,
            image_size=image_size,
        )

    if coordinate_transform == "direct":
        scale_x = image_width / xml_width
        scale_y = image_height / xml_height
        return [(x * scale_x, y * scale_y) for x, y in points]
    if coordinate_transform == "rotate_cw":
        scale_x = image_width / xml_height
        scale_y = image_height / xml_width
        return [((xml_height - y) * scale_x, x * scale_y) for x, y in points]
    if coordinate_transform == "rotate_ccw":
        scale_x = image_width / xml_height
        scale_y = image_height / xml_width
        return [(y * scale_x, (xml_width - x) * scale_y) for x, y in points]
    raise ValueError(f"unknown coordinate_transform: {coordinate_transform}")


def bbox_from_points(
    points: list[tuple[float, float]],
    *,
    xml_size: tuple[int, int],
    image_size: tuple[int, int],
    padding: int = 0,
    coordinate_transform: str = "auto",
) -> tuple[int, int, int, int]:
    if not points:
        raise ValueError("points must not be empty")
    image_width, image_height = image_size
    mapped = map_xml_points_to_image(
        points,
        xml_size=xml_size,
        image_size=image_size,
        coordinate_transform=coordinate_transform,
    )
    xs = [x for x, _ in mapped]
    ys = [y for _, y in mapped]
    left = max(0, int(min(xs) - padding))
    top = max(0, int(min(ys) - padding))
    right = min(image_width, int(max(xs) + padding))
    bottom = min(image_height, int(max(ys) + padding))
    if right <= left or bottom <= top:
        raise ValueError(f"invalid bbox from points: {(left, top, right, bottom)}")
    return left, top, right, bottom


def load_xml_page(path: str | Path) -> XmlPage:
    xml_path = Path(path)
    root = ET.parse(xml_path).getroot()
    width_text = root.get("width")
    height_text = root.get("height")
    if width_text is None or height_text is None:
        raise ValueError(f"missing page width/height in {xml_path}")

    chars: list[CharBox] = []
    for node in root.iter("char"):
        position = node.get("position")
        if not position:
            continue
        chars.append(
            CharBox(
                label=node.text.strip() if node.text is not None else None,
                points=parse_position(position),
                attrs=dict(node.attrib),
            )
        )

    return XmlPage(
        path=xml_path,
        image_id=root.get("id"),
        xml_size=(int(width_text), int(height_text)),
        chars=chars,
    )
