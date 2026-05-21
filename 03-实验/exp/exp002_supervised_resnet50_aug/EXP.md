# exp002_supervised_resnet50_aug

## 1. 目的

跑通第一版监督分类 baseline：从 `char_crops_384` 切分训练/验证集，训练 ResNet50，得到可追踪的本地验证指标和 checkpoint。

---

## 2. 背景

- 当前问题：已有 46,428 张单字 crop，但还没有训练/验证切分和模型训练结果。
- 上一个实验结论：`exp001` 已完成 crop 数据构建，坏框先跳过。
- 本实验为什么现在做：先建立端到端训练闭环，再讨论更强模型、预训练和少样本策略。

---

## 3. 设置

| 项 | 内容 |
|---|---|
| 数据版本 | `/root/autodl-tmp/datasets/processed/char_crops_384` |
| 训练/验证切分 | `splits/seed42_valid10`；单样本类别只进 train，每个可切分类至少留 1 张 train |
| 模型 | ResNet50 |
| 输入尺寸 | 224 |
| 预训练权重 | smoke 先用 `none`，确认链路后再试 `imagenet` |
| 关键增强 | pad 到正方形、轻微 affine、少量 blur |
| 优化器 | AdamW |
| 学习率 | 1e-3 |
| batch size | 64 |
| epoch | smoke 1；正式 baseline 5 |

---

## 4. 运行命令

先生成切分：

```bash
cd /root/autodl-tmp/projects/ancient-script-fewshot-recognition-
git pull --ff-only

PYTHONPATH=04-code python -m ancient_ocr.data.split_manifest \
  --manifest /root/autodl-tmp/datasets/processed/char_crops_384/manifest.jsonl \
  --output-dir /root/autodl-tmp/datasets/processed/char_crops_384/splits/seed42_valid10 \
  --valid-fraction 0.1 \
  --seed 42
```

smoke 训练：

```bash
PYTHONPATH=04-code python -m ancient_ocr.training.train_classifier \
  --data-root /root/autodl-tmp/datasets/processed/char_crops_384 \
  --train-split /root/autodl-tmp/datasets/processed/char_crops_384/splits/seed42_valid10/train.jsonl \
  --valid-split /root/autodl-tmp/datasets/processed/char_crops_384/splits/seed42_valid10/valid.jsonl \
  --label-map /root/autodl-tmp/datasets/processed/char_crops_384/splits/seed42_valid10/label_to_index.json \
  --output-dir /root/autodl-tmp/runs/exp002_supervised_resnet50_aug_smoke \
  --model resnet50 \
  --weights none \
  --image-size 224 \
  --batch-size 64 \
  --epochs 1 \
  --lr 0.001 \
  --num-workers 8 \
  --amp \
  --limit-train 5000 \
  --limit-valid 1000
```

正式 baseline：

```bash
PYTHONPATH=04-code python -m ancient_ocr.training.train_classifier \
  --data-root /root/autodl-tmp/datasets/processed/char_crops_384 \
  --train-split /root/autodl-tmp/datasets/processed/char_crops_384/splits/seed42_valid10/train.jsonl \
  --valid-split /root/autodl-tmp/datasets/processed/char_crops_384/splits/seed42_valid10/valid.jsonl \
  --label-map /root/autodl-tmp/datasets/processed/char_crops_384/splits/seed42_valid10/label_to_index.json \
  --output-dir /root/autodl-tmp/runs/exp002_supervised_resnet50_aug_full \
  --model resnet50 \
  --weights none \
  --image-size 224 \
  --batch-size 64 \
  --epochs 5 \
  --lr 0.001 \
  --num-workers 8 \
  --amp
```

---

## 5. 预期结果

- 最低通过标准：训练脚本完成 1 个 epoch，写出 `metrics.jsonl`、`config.json`、`last.pt`、`best.pt`。
- 失败时优先检查：split 路径、`label_index` 是否存在、torchvision 是否可用、显存是否够 batch size 64。

---

## 6. 实验结果

| 指标 | 数值 |
|---|---|
| split train records | 待跑 |
| split valid records | 待跑 |
| train loss | 待跑 |
| valid metric | 待跑 |
| online score | 无 |

---

## 7. 结论

**Verdict：Pending**

等待 AutoDL 开机后执行。

---

## 8. 产物

| 产物 | 路径 |
|---|---|
| split | `/root/autodl-tmp/datasets/processed/char_crops_384/splits/seed42_valid10` |
| smoke 日志/ckpt | `/root/autodl-tmp/runs/exp002_supervised_resnet50_aug_smoke` |
| full 日志/ckpt | `/root/autodl-tmp/runs/exp002_supervised_resnet50_aug_full` |
