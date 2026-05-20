from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from ancient_ocr.data.xml_crops import bbox_from_points, is_valid_label, load_xml_page


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def label_codepoints(label: str) -> str:
    return " ".join(f"U+{ord(char):04X}" for char in label)


def find_image_for_xml(xml_path: Path) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = xml_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _safe_char_id(xml_path: Path, index: int, attrs: dict[str, str]) -> str:
    char_id = attrs.get("id") or attrs.get("no") or str(index)
    keep = []
    for char in char_id:
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return f"{xml_path.stem}_{''.join(keep)}"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_crops(
    *,
    data_root: str | Path,
    output_dir: str | Path,
    padding: int = 16,
    limit: int | None = None,
) -> dict[str, int]:
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    Image.MAX_IMAGE_PIXELS = None

    records: list[dict[str, Any]] = []
    summary = {
        "xml_files": 0,
        "missing_images": 0,
        "invalid_labels": 0,
        "crop_errors": 0,
        "crops_written": 0,
    }

    for xml_path in sorted(data_root.rglob("*.xml")):
        if limit is not None and summary["crops_written"] >= limit:
            break
        summary["xml_files"] += 1
        image_path = find_image_for_xml(xml_path)
        if image_path is None:
            summary["missing_images"] += 1
            continue

        page = load_xml_page(xml_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_size = image.size
            for index, char in enumerate(page.chars, start=1):
                if limit is not None and summary["crops_written"] >= limit:
                    break
                if not is_valid_label(char.label):
                    summary["invalid_labels"] += 1
                    continue
                try:
                    bbox = bbox_from_points(
                        char.points,
                        xml_size=page.xml_size,
                        image_size=image_size,
                        padding=padding,
                    )
                    crop = image.crop(bbox)
                    crop_name = f"{_safe_char_id(xml_path, index, char.attrs)}.png"
                    crop_path = crop_dir / crop_name
                    crop.save(crop_path)
                except Exception:
                    summary["crop_errors"] += 1
                    continue

                records.append(
                    {
                        "crop_path": str(crop_path.relative_to(output_dir)),
                        "image_path": str(image_path.relative_to(data_root)),
                        "xml_path": str(xml_path.relative_to(data_root)),
                        "label": char.label,
                        "label_codepoints": label_codepoints(char.label or ""),
                        "bbox": list(bbox),
                        "xml_size": list(page.xml_size),
                        "image_size": list(image_size),
                    }
                )
                summary["crops_written"] += 1

    _write_jsonl(output_dir / "manifest.jsonl", records)
    _write_jsonl(output_dir / "summary.jsonl", [summary])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract single-character crops from XML annotations.")
    parser.add_argument("--data-root", required=True, help="Path to extracted train directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for crops and manifest.jsonl.")
    parser.add_argument("--padding", type=int, default=16, help="Padding around each XML bbox in pixels.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of crops to write.")
    args = parser.parse_args()

    summary = extract_crops(
        data_root=args.data_root,
        output_dir=args.output_dir,
        padding=args.padding,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
