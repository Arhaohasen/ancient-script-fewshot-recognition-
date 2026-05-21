from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any

from PIL import Image, ImageOps


def read_jsonl(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def load_index_to_label(path: str | Path) -> list[str]:
    label_to_index = json.loads(Path(path).read_text(encoding="utf-8"))
    index_to_label = [None] * len(label_to_index)
    for label, index in label_to_index.items():
        index_to_label[index] = label
    if any(label is None for label in index_to_label):
        raise ValueError(f"label map has missing indices: {path}")
    return [str(label) for label in index_to_label]


def build_run_config(
    *,
    data_root: str,
    train_split: str,
    valid_split: str,
    label_map: str,
    output_dir: str,
    model: str,
    weights: str,
    image_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    num_workers: int,
    amp: bool,
    limit_train: int | None,
    limit_valid: int | None,
) -> dict[str, Any]:
    return {
        "data_root": data_root,
        "train_split": train_split,
        "valid_split": valid_split,
        "label_map": label_map,
        "output_dir": output_dir,
        "model": model,
        "weights": weights,
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "num_workers": num_workers,
        "amp": amp,
        "limit_train": limit_train,
        "limit_valid": limit_valid,
    }


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_transforms(image_size: int):
    from torchvision import transforms

    return {
        "train": transforms.Compose(
            [
                transforms.Lambda(lambda image: ImageOps.pad(image.convert("RGB"), (image_size, image_size), color="white")),
                transforms.RandomApply(
                    [transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.92, 1.08))],
                    p=0.7,
                ),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Lambda(lambda image: ImageOps.pad(image.convert("RGB"), (image_size, image_size), color="white")),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
    }


class CropJsonlDataset:
    def __init__(self, *, data_root: str | Path, records: list[dict[str, Any]], transform) -> None:
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __len__(self_inner) -> int:
                return len(records)

            def __getitem__(self_inner, index: int):
                record = records[index]
                path = Path(data_root) / str(record["crop_path"])
                with Image.open(path) as image:
                    tensor = transform(image)
                return tensor, int(record["label_index"])

        self.dataset = _Dataset()


def _build_model(name: str, *, num_classes: int, weights: str):
    import torch.nn as nn
    from torchvision import models

    if name != "resnet50":
        raise ValueError(f"unsupported model: {name}")

    if weights == "imagenet":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif weights == "none":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"unsupported weights: {weights}")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_classifier(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    _seed_everything(int(config["seed"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    index_to_label = load_index_to_label(config["label_map"])
    transforms = _build_transforms(int(config["image_size"]))
    train_records = read_jsonl(config["train_split"], limit=config["limit_train"])
    valid_records = read_jsonl(config["valid_split"], limit=config["limit_valid"])
    train_dataset = CropJsonlDataset(
        data_root=config["data_root"],
        records=train_records,
        transform=transforms["train"],
    ).dataset
    valid_dataset = CropJsonlDataset(
        data_root=config["data_root"],
        records=valid_records,
        transform=transforms["valid"],
    ).dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = _build_model(config["model"], num_classes=len(index_to_label), weights=config["weights"]).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["amp"]) and device.type == "cuda")

    best_valid_acc = -1.0
    best_metrics: dict[str, Any] = {}
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    for epoch in range(1, int(config["epochs"]) + 1):
        started = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(config["amp"]) and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.size(0)
            train_loss += float(loss.item()) * batch_size
            train_correct += int((logits.argmax(dim=1) == targets).sum().item())
            train_total += batch_size

        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for images, targets in valid_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, targets)
                batch_size = targets.size(0)
                valid_loss += float(loss.item()) * batch_size
                valid_correct += int((logits.argmax(dim=1) == targets).sum().item())
                valid_total += batch_size

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_total),
            "train_acc": train_correct / max(1, train_total),
            "valid_loss": valid_loss / max(1, valid_total),
            "valid_acc": valid_correct / max(1, valid_total),
            "train_total": train_total,
            "valid_total": valid_total,
            "seconds": round(time.time() - started, 3),
        }
        _write_jsonl(metrics_path, metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
            "index_to_label": index_to_label,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if metrics["valid_acc"] > best_valid_acc:
            best_valid_acc = metrics["valid_acc"]
            best_metrics = metrics
            torch.save(checkpoint, output_dir / "best.pt")

    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised crop classifier baseline.")
    parser.add_argument("--data-root", required=True, help="Directory containing crops/ and manifest-derived paths.")
    parser.add_argument("--train-split", required=True, help="Path to train.jsonl.")
    parser.add_argument("--valid-split", required=True, help="Path to valid.jsonl.")
    parser.add_argument("--label-map", required=True, help="Path to label_to_index.json.")
    parser.add_argument("--output-dir", required=True, help="Run output directory.")
    parser.add_argument("--model", default="resnet50", choices=["resnet50"])
    parser.add_argument("--weights", default="none", choices=["none", "imagenet"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-valid", type=int, default=None)
    args = parser.parse_args()

    config = build_run_config(
        data_root=args.data_root,
        train_split=args.train_split,
        valid_split=args.valid_split,
        label_map=args.label_map,
        output_dir=args.output_dir,
        model=args.model,
        weights=args.weights,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
        amp=args.amp,
        limit_train=args.limit_train,
        limit_valid=args.limit_valid,
    )
    best_metrics = train_classifier(config)
    print(json.dumps({"best": best_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
