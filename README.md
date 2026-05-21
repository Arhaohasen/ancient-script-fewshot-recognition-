# 少样本跨域古文字识别比赛

> **当前状态（2026-05-22）**：数据已解压统计，`exp002_supervised_resnet50_aug` 已有 ResNet50 checkpoint，推理入口已按赛方 schema 输出 `/saisresult/prediction.json`。当前优先目标是构建并 smoke 可提交 Docker 镜像。
>
> **新会话开工先读**：`AGENTS.md`（项目长期记忆）→ `00-赛题资料/赛题分析.md` → `00-赛题资料/数据说明.md` → `03-实验/JOURNAL.md`。

---

## 一句话理解

这是一个“少样本 + 跨域 + 古文字识别”任务：训练集有限，测试域可能与训练域不同，需要模型既能识别古文字形态，又能抵抗字体、载体、时代、采集条件带来的分布变化。

---

## 当前优先级

| 优先级 | 事项 | 目标 |
|---|---|---|
| P0 | 赛题与数据理解 | 明确数据格式、类别空间、评价指标、提交格式 |
| P0 | Baseline | 跑通训练、验证、推理、提交全链路 |
| P1 | 跨域策略 | 做数据增强、预训练/微调、原型网络或度量学习对比 |
| P1 | 少样本策略 | 建立 episodic training、prompt/adapter 或 class prototype 实验 |
| P2 | 模型创新 | 在可复现实验基础上再做结构或训练策略改进 |

---

## 目录结构

```text
少样本跨域古文字识别/
├── AGENTS.md                 # 项目长期记忆：目标、约束、当前决策
├── README.md                 # 项目入口
├── 00-赛题资料/              # 赛题、数据、提交规则
├── 01-学习路线/              # 古文字识别、少样本学习、跨域泛化基础
├── 02-文献池/                # 论文索引与笔记模板
├── 03-实验/                  # 实验日志、机器 ledger、EXP 模板
├── 04-code/                  # 代码区：数据、模型、训练、推理、工具
├── 05-submission/            # 提交文件与提交记录
└── archive/                  # 历史材料与不再进入主上下文的归档
```

---

## 实验管理三件套

| 文件 | 作用 |
|---|---|
| `03-实验/JOURNAL.md` | 人读实验日志：每次实验写结论、原因、下一步 |
| `03-实验/runs.jsonl` | 机器读实验流水账：每行一个 run |
| `03-实验/EXP.md.template` | 每个实验目录的记录模板 |

每次跑实验前先复制 `EXP.md.template` 到 `03-实验/exp/expNNN_xxx/EXP.md`，跑完后同步更新 `JOURNAL.md` 和 `runs.jsonl`。

---

## 下一步

1. 将 `/root/autodl-tmp/runs/exp002_supervised_resnet50_imagenet_filtered_full/best.pt` 复制为 `models/best.pt`。
2. 构建 `ancient-script-submit:exp002` Docker 镜像并用 `/saisdata/13/eval/images/` 结构 smoke。
3. 若镜像链路无误，提交一次流程 baseline；随后再优化检测器和识别模型。

---

## 初赛镜像提交说明

### 运行环境

当前提交镜像基于：

| 项 | 版本 |
|---|---|
| Base image | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` |
| OS | Ubuntu / Linux runtime from PyTorch image |
| Python | 3.11 |
| CUDA | 12.4 |
| cuDNN | 9 |
| PyTorch | 2.5.1 |
| 关键依赖 | `numpy>=1.24`, `Pillow>=10.0`, `torchvision` from base image |

本地在 Apple Silicon Mac 构建时需要指定 `--platform linux/amd64`，以适配赛方 Linux + NVIDIA GPU 环境。

### 推理入口

镜像启动后自动执行：

```text
/app/run.sh
```

默认路径：

| 路径 | 用途 |
|---|---|
| `/saisdata/13/eval/images/` | 读取评测 PNG 图片 |
| `/app/04-code/` | 推理代码 |
| `/app/best.pt` | 模型权重 |
| `/saisresult/prediction.json` | 输出结果 |

输出 JSON 为 UTF-8 编码，顶层结构为 `{image_id: [{"bbox": [x, y, w, h], "text": "..."}]}`。若某张图没有检测结果，会保留该 `image_id` 并写空列表。

### 模型方案

当前镜像是流程 baseline：

1. 用传统前景连通域方法从整页图片中生成候选字符框。
2. 将候选框裁剪并缩放到分类器输入尺寸。
3. 使用 ImageNet 预训练 ResNet50 微调得到的古文字字符分类器识别单字。
4. 将检测框和识别文本按赛方格式写入 `prediction.json`。

当前候选 checkpoint：

```text
/app/best.pt
```

对应训练产物：

```text
/root/autodl-tmp/runs/exp002_supervised_resnet50_imagenet_filtered_full/best.pt
```

训练数据来自赛方初赛训练集。训练时过滤 `None`、空标签、`ZH-` / `ZHFD-` 等无法映射到有效 Unicode 码点的占位标签。

### 训练方式

- 从赛方 XML 标注中解析字符框和 Unicode 文本标签。
- 裁剪字符图，构建 `char_crops_384` 数据集。
- 使用确定性 seed 切分训练/验证集。
- 用 ImageNet 预训练 ResNet50 做监督微调。
- 当前记录的内部验证集 top-1 accuracy：32.18%。

### 外部数据

当前 baseline 未使用额外标注数据。模型初始化使用 torchvision/ImageNet 预训练权重；正式推理不依赖公网下载模型、数据或代码，所需代码和权重均打包在 Docker 镜像内。
