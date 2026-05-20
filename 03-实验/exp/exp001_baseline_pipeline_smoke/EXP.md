# exp001_baseline_pipeline_smoke

## 1. 目的

跑通从原始 XML/图片到单字 crop 数据集的第一段 pipeline，并为后续监督分类 baseline 准备稳定输入。

---

## 2. 背景

- 当前问题：原始数据是整页图片 + XML 单字标注，不能直接喂给分类模型。
- 上一个实验结论：尚无模型实验；已确认 `train.tar` 完整下载并解压。
- 本实验为什么现在做：先把数据读取、标签过滤、坐标映射和裁剪产物跑通，后续模型训练才有可信输入。

---

## 3. 设置

| 项 | 内容 |
|---|---|
| 数据版本 | `/root/autodl-tmp/datasets/train/` |
| 训练/验证切分 | 待定，本实验先只构建 crop 数据 |
| 模型 | 无 |
| 输入尺寸 | crop 最长边 384，训练时再 resize |
| 预训练权重 | 无 |
| 关键增强 | 无 |
| 优化器 | 无 |
| 学习率 | 无 |
| batch size | 无 |
| epoch | 无 |

---

## 4. 运行命令

```bash
cd /root/autodl-tmp/projects/ancient-script-fewshot-recognition-

PYTHONPATH=04-code python -m ancient_ocr.data.build_crops \
  --data-root /root/autodl-tmp/datasets/train \
  --output-dir /root/autodl-tmp/datasets/processed/char_crops_384 \
  --padding 24 \
  --max-side 384
```

后台运行：

```bash
screen -dmS crops384 bash -lc 'cd /root/autodl-tmp/projects/ancient-script-fewshot-recognition- && PYTHONPATH=04-code python -m ancient_ocr.data.build_crops --data-root /root/autodl-tmp/datasets/train --output-dir /root/autodl-tmp/datasets/processed/char_crops_384 --padding 24 --max-side 384 > /root/autodl-tmp/datasets/processed/char_crops_384.log 2>&1'
```

---

## 5. 预期结果

- 最低通过标准：生成 `crops/`、`manifest.jsonl`、`errors.jsonl`、`summary.jsonl`；成功 crop 数接近 61,248。
- 失败时优先检查：XML 编码、`position` 格式、XML 坐标系与图片尺寸映射、坏框是否集中在特定来源。

---

## 6. 实验结果

| 指标 | 数值 |
|---|---|
| smoke crops | 100 |
| smoke crop errors | 3 |
| smoke output size | 14 MB |
| full crops | 生成中 |
| full crop errors | 生成中 |

---

## 7. 结论

**Verdict：Pending**

smoke 结果显示裁剪链路可用，`--max-side 384` 能把中间数据体积从原始 crop 的约 63 MB/100 张降到约 14 MB/100 张。等待全量裁剪完成后补最终统计。

---

## 8. 产物

| 产物 | 路径 |
|---|---|
| smoke 数据 | `/root/autodl-tmp/datasets/processed/char_crops_smoke_v3_384` |
| full 数据 | `/root/autodl-tmp/datasets/processed/char_crops_384` |
| full 日志 | `/root/autodl-tmp/datasets/processed/char_crops_384.log` |
| contact sheet | `/root/autodl-tmp/datasets/char_crops_smoke_v3_384_contact.jpg` |
