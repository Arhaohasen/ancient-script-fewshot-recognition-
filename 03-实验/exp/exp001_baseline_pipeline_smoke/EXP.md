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
| full crops | 46,428 |
| full crop errors | 14,820 |
| invalid labels | 23 |
| full output size | 约 3.8 GB |
| unique labels | 3,530 |
| singleton labels | 1,757 |
| multi-codepoint labels | 532 |

---

## 7. 结论

**Verdict：Positive**

裁剪链路可用，已生成第一版 baseline 输入数据集。`--max-side 384` 能把中间数据体积控制在约 3.8 GB，适合后续训练读取。

需要注意：14,820 个标注框裁剪失败，均为 `invalid bbox from points`。复查发现绝大多数失败框在 direct、顺时针、逆时针三种坐标映射下都无法得到有效 bbox，初步判断不是简单旋转规则问题，而是原始标注坐标落在当前图片外或图片版本不匹配。第一版模型先使用 46,428 张成功 crop，坏框后续再作为数据清洗专题处理。

---

## 8. 产物

| 产物 | 路径 |
|---|---|
| smoke 数据 | `/root/autodl-tmp/datasets/processed/char_crops_smoke_v3_384` |
| full 数据 | `/root/autodl-tmp/datasets/processed/char_crops_384` |
| full 日志 | `/root/autodl-tmp/datasets/processed/char_crops_384.log` |
| 类别统计 | `/root/autodl-tmp/datasets/processed/char_crops_384/stats` |
| contact sheet | `/root/autodl-tmp/datasets/char_crops_smoke_v3_384_contact.jpg` |
