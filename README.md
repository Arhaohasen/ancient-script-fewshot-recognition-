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
