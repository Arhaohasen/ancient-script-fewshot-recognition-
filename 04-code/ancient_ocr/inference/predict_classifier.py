from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class InputImage:
    path: Path
    relative_path: str
    image_id: str


def resolve_image_dir(input_dir: str | Path) -> Path:
    root = Path(input_dir)
    sais_eval_dir = root / "13" / "eval" / "images"
    if sais_eval_dir.exists():
        return sais_eval_dir
    return root


def find_input_images(input_dir: str | Path) -> list[InputImage]:
    root = resolve_image_dir(input_dir)
    images: list[InputImage] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(
                InputImage(
                    path=path,
                    relative_path=path.relative_to(root).as_posix(),
                    image_id=path.stem,
                )
            )
    return images


def build_prediction_payload(
    *,
    predictions: Iterable[tuple[str, list[dict[str, Any]]]],
    schema: str,
) -> Any:
    items = list(predictions)
    if schema == "sais":
        return {image_id: detections for image_id, detections in items}
    raise ValueError(f"unsupported prediction schema: {schema}")


def write_prediction_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    json.loads(output_path.read_text(encoding="utf-8"))


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
    median = float(np.median(gray))
    q20 = float(np.percentile(gray, 20))
    q80 = float(np.percentile(gray, 80))
    if median < 128:
        mask = gray >= max(q80, median + 20)
    else:
        mask = gray <= min(q20, median - 20)

    ratio = float(mask.mean())
    if ratio < 0.001 or ratio > 0.55:
        diff = np.abs(gray.astype(np.int16) - int(median))
        threshold = max(20, float(np.percentile(diff, 85)))
        mask = diff >= threshold
    return mask


def _connected_components_cv2(mask: np.ndarray) -> list[tuple[int, int, int, int, int]] | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return None

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    components: list[tuple[int, int, int, int, int]] = []
    for index in range(1, count):
        left, top, width, height, area = stats[index]
        components.append((int(left), int(top), int(left + width), int(top + height), int(area)))
    return components


def _connected_components_numpy(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        stack = [(start_x, start_y)]
        visited[start_y, start_x] = True
        min_x = max_x = start_x
        min_y = max_y = start_y
        area = 0
        while stack:
            x, y = stack.pop()
            area += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for next_x, next_y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= next_x < width and 0 <= next_y < height and mask[next_y, next_x] and not visited[next_y, next_x]:
                    visited[next_y, next_x] = True
                    stack.append((next_x, next_y))
        components.append((min_x, min_y, max_x + 1, max_y + 1, area))
    return components


def _connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    return _connected_components_cv2(mask) or _connected_components_numpy(mask)


def detect_candidate_boxes(
    image: Image.Image,
    *,
    min_area: int = 64,
    max_boxes: int = 64,
    pad: int = 4,
    max_detection_side: int = 800,
) -> list[list[int]]:
    original_width, original_height = image.size
    scale = 1.0
    working = image.convert("L")
    longest_side = max(original_width, original_height)
    if longest_side > max_detection_side:
        scale = max_detection_side / longest_side
        working = working.resize(
            (max(1, round(original_width * scale)), max(1, round(original_height * scale))),
            Image.Resampling.BILINEAR,
        )

    gray = np.array(working)
    mask = _foreground_mask(gray)
    components = _connected_components(mask)
    scaled_min_area = max(4, round(min_area * scale * scale))
    image_area = gray.shape[0] * gray.shape[1]
    candidates: list[tuple[int, int, int, int, int]] = []
    for left, top, right, bottom, area in components:
        width = right - left
        height = bottom - top
        box_area = width * height
        if area < scaled_min_area:
            continue
        if area > image_area * 0.25:
            continue
        if box_area > image_area * 0.08:
            continue
        if width > gray.shape[1] * 0.65 or height > gray.shape[0] * 0.65:
            continue
        if width < 3 or height < 3:
            continue
        candidates.append((left, top, right, bottom, area))

    candidates.sort(key=lambda item: (item[1], item[0], -item[4]))
    boxes: list[list[int]] = []
    for left, top, right, bottom, _ in candidates[:max_boxes]:
        x1 = max(0, int((left - pad) / scale))
        y1 = max(0, int((top - pad) / scale))
        x2 = min(original_width, int((right + pad) / scale))
        y2 = min(original_height, int((bottom + pad) / scale))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2 - x1, y2 - y1])

    if not boxes:
        boxes.append([0, 0, original_width, original_height])
    return boxes


def _build_valid_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Lambda(lambda image: ImageOps.pad(image.convert("RGB"), (image_size, image_size), color="white")),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _build_model(config: dict[str, Any], *, num_classes: int):
    from ancient_ocr.training.train_classifier import _build_model

    return _build_model(
        config.get("model", "resnet50"),
        num_classes=num_classes,
        weights="none",
    )


def predict_images(
    *,
    input_dir: str | Path,
    checkpoint_path: str | Path,
    batch_size: int = 128,
    max_boxes: int = 64,
    max_detection_side: int = 800,
    progress_interval: int = 25,
) -> list[tuple[str, list[dict[str, Any]]]]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    index_to_label = [str(label) for label in checkpoint["index_to_label"]]
    config = checkpoint.get("config", {})
    image_size = int(config.get("image_size", 224))
    transform = _build_valid_transform(image_size)
    images = find_input_images(input_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(config, num_classes=len(index_to_label))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    predictions: list[tuple[str, list[dict[str, Any]]]] = []
    with torch.no_grad():
        total_images = len(images)
        for image_index, item in enumerate(images, start=1):
            with Image.open(item.path) as image:
                image = image.convert("RGB")
                boxes = detect_candidate_boxes(
                    image,
                    max_boxes=max_boxes,
                    max_detection_side=max_detection_side,
                )
                detections: list[dict[str, Any]] = []
                for start in range(0, len(boxes), batch_size):
                    chunk = boxes[start : start + batch_size]
                    tensors = []
                    for x, y, width, height in chunk:
                        crop = image.crop((x, y, x + width, y + height))
                        tensors.append(transform(crop))
                    batch = torch.stack(tensors).to(device, non_blocking=True)
                    indices = model(batch).argmax(dim=1).cpu().tolist()
                    detections.extend(
                        {
                            "bbox": [int(x), int(y), int(width), int(height)],
                            "text": index_to_label[index],
                        }
                        for (x, y, width, height), index in zip(chunk, indices)
                    )
                predictions.append((item.image_id, detections))
            if progress_interval > 0 and (image_index == 1 or image_index % progress_interval == 0 or image_index == total_images):
                print(
                    json.dumps(
                        {
                            "processed": image_index,
                            "total": total_images,
                            "last_image_id": item.image_id,
                            "detections": len(detections),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return predictions


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run classifier inference and write prediction.json.")
    parser.add_argument("--input-dir", default="/saisdata", help="Input data directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--output", default="/saisresult/prediction.json", help="Output prediction.json path.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-boxes", type=int, default=64, help="Maximum candidate boxes per image.")
    parser.add_argument(
        "--max-detection-side",
        type=int,
        default=800,
        help="Resize the longest side to this value for foreground connected-component detection.",
    )
    parser.add_argument("--progress-interval", type=int, default=25, help="Print progress every N images. Use 0 to disable.")
    parser.add_argument(
        "--schema",
        default="sais",
        choices=["sais"],
        help="Output schema required by the competition: image_id -> list of bbox/text detections.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    predictions = predict_images(
        input_dir=args.input_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        max_boxes=args.max_boxes,
        max_detection_side=args.max_detection_side,
        progress_interval=args.progress_interval,
    )
    payload = build_prediction_payload(predictions=predictions, schema=args.schema)
    write_prediction_json(args.output, payload)
    print(json.dumps({"images": len(predictions), "output": args.output, "schema": args.schema}, ensure_ascii=False))


if __name__ == "__main__":
    main()
