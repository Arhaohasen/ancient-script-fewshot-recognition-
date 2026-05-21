# 实验日志

> 人读日志。每次实验结束后写结论，不写流水账。推荐格式：结论先行，再写证据、原因、下一步。

---

## 2026-05-10 — 项目启动

**Verdict：Pending**

已建立比赛优先版项目结构。当前还没有下载和解压数据，因此暂不开始模型实验。下一步是完成 `train.tar` 下载、解压、数据结构检查，并创建 `exp001_baseline_pipeline_smoke`。

---

## 2026-05-21 — exp001_baseline_pipeline_smoke

**Verdict：Positive**

已完成第一版单字裁剪数据集：`--max-side 384` 版本写出 46,428 张成功 crop，输出约 3.8 GB，可作为监督分类 baseline 的默认输入。

### 设置

- 数据：`/root/autodl-tmp/datasets/train/`
- 模型：无，本阶段只构建 baseline 输入数据
- 关键参数：`--padding 24 --max-side 384`
- 验证集：暂未切分

### 结果

- 本地指标：15 个数据处理单元测试通过
- 线上分数：无
- 训练现象：无
- 数据产物：46,428 crops，3,530 类，1,757 个单样本类别
- 异常：23 个无效标签，14,820 个裁剪失败框

### 解释

- 裁剪链路可行，但 14,820 个原始 bbox 会映射到图片外，当前先记录到 `errors.jsonl` 并跳过。
- 复查失败框后，14,777 个在 direct、顺时针、逆时针三种坐标映射下都无法得到有效 bbox，初步判断不是简单旋转规则问题。
- 384 crop 比原始 crop 更省磁盘和 I/O，适合作为第一版监督分类 baseline 的输入。

### 下一步

- 基于 `manifest.jsonl` 做类别长尾统计和验证集切分。
- 启动 ResNet/ConvNeXt 监督分类 smoke。

---

## 2026-05-22 — exp002_supervised_resnet50_aug

**Verdict：Pending**

已补训练/验证切分脚本和 ResNet50 训练入口，等待 AutoDL 开机后跑真实数据 smoke。

### 设置

- 数据：`/root/autodl-tmp/datasets/processed/char_crops_384`
- 模型：ResNet50
- 关键超参：224 输入，batch size 64，AdamW，lr 1e-3
- 验证集：计划使用 `splits/seed42_valid10`

### 结果

- 本地指标：21 个单元测试通过
- 线上分数：无
- 训练现象：待跑

### 解释

- 单样本类别只进训练集，避免验证集中出现训练完全没见过的类别。
- smoke 先使用 `weights=none`，减少首次运行对外网下载预训练权重的依赖。

### 下一步

- AutoDL 开机后拉取最新代码。
- 生成 split 并跑 1 epoch smoke。
- 根据 smoke 速度和显存，决定是否跑 5 epoch full baseline。

---

## 日志模板

```markdown
## YYYY-MM-DD — expNNN_shortname

**Verdict：Positive / Negative / Pending**

一句话结论。

### 设置

- 数据：
- 模型：
- 关键超参：
- 验证集：

### 结果

- 本地指标：
- 线上分数：
- 训练现象：

### 解释

- 为什么涨/跌/无效：
- 是否可信：

### 下一步

- 
```
