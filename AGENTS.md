# AGENTS.md — 少样本跨域古文字识别项目长期记忆

> **定位**：这是项目的长期记忆。每次新会话先读本文件，避免重复讨论背景、目标和规则。
>
> **维护节奏**：每完成一个里程碑、拿到一次有效提交分、发现一个数据坑或做出方向决策，就更新一次。

---

## 0. 一句话现状

**2026-05-22**：项目骨架和数据处理/训练/推理入口代码已推送到 GitHub 私有仓库，AutoDL 训练机已配置 1 张 RTX 4090。`train.tar` 已下载并解压，原始数据结构已统计；`exp001_baseline_pipeline_smoke` 已完成，`/root/autodl-tmp/datasets/processed/char_crops_384` 生成 46,428 张成功 crop。`exp002_supervised_resnet50_aug` 已跑通：过滤 `ZHFD-...` 等占位标签后，ImageNet 预训练 ResNet50 5 epoch 在内部验证集达到 32.18% top-1 accuracy。AutoDL `/app` 已准备好 `run.sh`、`04-code/`、`best.pt`，并用默认 `/saisdata -> /saisresult/prediction.json` 路径做过 8 张图片推理 smoke。下一步是确认 `prediction.json` 内部 schema 和测试样本 ID 来源。

---

## 1. 项目目标

| 优先级 | 目标 | 验收口径 |
|---|---|---|
| P0 | 跑通 baseline | 能从原始数据生成一次合法提交 |
| P0 | 建立实验纪律 | 每个实验都有 `EXP.md`、`JOURNAL.md` 结论和 `runs.jsonl` 记录 |
| P1 | 提升少样本识别能力 | 对小样本类别有稳定验证集提升 |
| P1 | 提升跨域泛化能力 | 在不同来源/风格/噪声条件下不过拟合训练域 |
| P2 | 整理比赛复盘 | 形成可复用的方法总结和后续论文/项目材料 |

---

## 2. 当前判断

### 2.1 任务关键词

- **古文字识别**：图像分类或图像到字符/类别的识别任务，重点在形态差异细、噪声大、类别相似。
- **少样本学习**：每个类别标注样本少，不能只依赖大规模监督训练。
- **跨域泛化**：训练数据和测试数据可能来自不同字形、材质、拓片/扫描/拍照条件或历史时期。

### 2.2 优先技术路线

先不急着做复杂创新，按三层推进：

1. **流程 baseline**：数据读取、训练、验证、推理、提交。
2. **强监督 baseline**：ResNet / ConvNeXt / ViT（Vision Transformer，视觉 Transformer）微调。
3. **少样本与跨域增强**：度量学习、原型分类、强数据增强、自监督预训练、领域泛化策略。

---

## 3. 工作规则

1. **先数据后模型**：没看清数据结构、标签分布和提交格式前，不做模型复杂化。
2. **先跑通后涨分**：第一个实验只追求端到端正确，不追求 leaderboard 高分。
3. **每次实验必须留痕**：实验开始前写 `EXP.md`，结束后更新 `JOURNAL.md` 和 `runs.jsonl`。
4. **大文件不入库**：原始数据、模型 checkpoint、日志、提交压缩包默认不进 git。
5. **不盲目堆模型**：优先找数据切分、增强、预训练和验证集设计的问题。

---

## 4. 当前阻塞项

| 阻塞项 | 状态 | 下一步 |
|---|---|---|
| 训练数据 | 已下载并解压 | 保留原始压缩包，使用 `/root/autodl-tmp/datasets/train/` |
| 数据格式 | 已完成首轮统计 | 使用 `char_crops_384` 跑 baseline，坏框后续再清洗 |
| 评价指标 | 待确认 | 从赛题页面或提交反馈确认 |
| 提交格式 | 部分确认 | 镜像运行路径已确认，`prediction.json` 内部 schema 待确认 |

---

## 4.1 已确认赛方规则

- `prediction.json` 提交文件必须使用 UTF-8 编码。
- 镜像提交入口必须为 `/app/run.sh`。
- 选手程序必须从 `/saisdata` 读取输入数据集。
- 输出结果必须写到 `/saisresult/prediction.json`。
- 镜像提交截止时间为 2026-05-27 14:00:00。
- UTF-8 可以表示全部 Unicode 字符，不存在“某些字符只能用 UTF-16 编码”的问题。
- XML 标签或古文字在编辑器中显示成乱码/方块，通常是本地字体或编辑器缺字导致；只要程序能正确解析为有效 Unicode 码点，就属于正常数据，可用于训练。
- `ZH-` 开头字符串、`None` 等属于标注时无法映射到标准 Unicode 的占位符或异常值，不是有效 Unicode 码点，应直接过滤。
- 判断标签是否有效时，以程序读取后的 Unicode 码点是否有效为准，不以屏幕是否能正常显示为准。

---

## 5. 实验命名约定

```text
expNNN_{stage}_{method}_{shortname}/
```

示例：

- `exp001_baseline_pipeline_smoke`
- `exp002_supervised_resnet50_aug`
- `exp003_fewshot_proto_vit`
- `exp004_domain_generalization_strong_aug`

---

## 6. 红线

| # | 不要做的事 | 原因 |
|---|---|---|
| 1 | 还没读数据就写复杂模型 | 容易解决错问题 |
| 2 | 没有验证集就根据线上分盲调 | 每天提交次数有限，且无法定位问题 |
| 3 | 只记录“跑了什么”不记录“为什么” | 复盘和接力会断 |
| 4 | 把 37 GB 原始数据或 checkpoint 放进 git | 仓库会不可用 |
| 5 | 看到少样本就直接上花哨方法 | 很多时候数据增强和预训练 baseline 更稳 |

---

## 7. 下一次开工清单

1. 确认 `prediction.json` 内部 schema：列表/字典/字段名，以及测试样本 ID 来源。
2. 补 `/app/run.sh` 兼容入口和推理脚本：读取 checkpoint、从 `/saisdata` 预测、输出 UTF-8 `/saisresult/prediction.json`。
3. 如果暂无测试集，先基于 valid split 做离线推理 smoke，确保 label 解码正确。
4. 基于 ImageNet ResNet50 baseline 尝试 ConvNeXt/更强增强/类别重采样。
5. 单独分析 14,820 个坏框是否能通过图片版本、坐标系或标注规则修复。
