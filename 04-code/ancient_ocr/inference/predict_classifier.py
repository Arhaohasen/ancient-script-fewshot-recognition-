from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class InputImage:
    path: Path
    relative_path: str


def find_input_images(input_dir: str | Path) -> list[InputImage]:
    root = Path(input_dir)
    images: list[InputImage] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(InputImage(path=path, relative_path=path.relative_to(root).as_posix()))
    return images


def build_prediction_payload(
    *,
    predictions: Iterable[tuple[str, str]],
    schema: str,
) -> Any:
    items = list(predictions)
    if schema == "mapping":
        return {image_id: label for image_id, label in items}
    if schema == "records":
        return [{"image_id": image_id, "prediction": label} for image_id, label in items]
    raise ValueError(f"unsupported prediction schema: {schema}")


def write_prediction_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    json.loads(output_path.read_text(encoding="utf-8"))


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
) -> list[tuple[str, str]]:
    import torch
    from torch.utils.data import DataLoader, Dataset

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    index_to_label = [str(label) for label in checkpoint["index_to_label"]]
    config = checkpoint.get("config", {})
    image_size = int(config.get("image_size", 224))
    transform = _build_valid_transform(image_size)
    images = find_input_images(input_dir)

    class _ImageDataset(Dataset):
        def __len__(self) -> int:
            return len(images)

        def __getitem__(self, index: int):
            item = images[index]
            with Image.open(item.path) as image:
                tensor = transform(image)
            return tensor, item.relative_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(config, num_classes=len(index_to_label))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    loader = DataLoader(
        _ImageDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    predictions: list[tuple[str, str]] = []
    with torch.no_grad():
        for batch, relative_paths in loader:
            logits = model(batch.to(device, non_blocking=True))
            indices = logits.argmax(dim=1).cpu().tolist()
            predictions.extend((relative_path, index_to_label[index]) for relative_path, index in zip(relative_paths, indices))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classifier inference and write prediction.json.")
    parser.add_argument("--input-dir", default="/saisdata", help="Input data directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--output", default="/saisresult/prediction.json", help="Output prediction.json path.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--schema",
        default="mapping",
        choices=["mapping", "records"],
        help="Temporary output schema until official prediction.json fields are confirmed.",
    )
    args = parser.parse_args()

    predictions = predict_images(
        input_dir=args.input_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
    )
    payload = build_prediction_payload(predictions=predictions, schema=args.schema)
    write_prediction_json(args.output, payload)
    print(json.dumps({"images": len(predictions), "output": args.output, "schema": args.schema}, ensure_ascii=False))


if __name__ == "__main__":
    main()
