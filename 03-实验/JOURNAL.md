# 实验日志

> 人读日志。每次实验结束后写结论，不写流水账。推荐格式：结论先行，再写证据、原因、下一步。

---

## 2026-05-10 — 项目启动

**Verdict：Pending**

已建立比赛优先版项目结构。当前还没有下载和解压数据，因此暂不开始模型实验。下一步是完成 `train.tar` 下载、解压、数据结构检查，并创建 `exp001_baseline_pipeline_smoke`。

---

## 2026-05-21 — exp001_baseline_pipeline_smoke

**Verdict：Pending**

已跑通 100 张单字裁剪 smoke：`--max-side 384` 版本成功写出 100 张 crop，记录 3 个坏框，输出约 14 MB。全量裁剪任务已在 AutoDL 后台运行，目标产物是 `/root/autodl-tmp/datasets/processed/char_crops_384`。

### 设置

- 数据：`/root/autodl-tmp/datasets/train/`
- 模型：无，本阶段只构建 baseline 输入数据
- 关键参数：`--padding 24 --max-side 384`
- 验证集：暂未切分

### 结果

- 本地指标：13 个数据处理单元测试通过
- 线上分数：无
- 训练现象：无

### 解释

- 裁剪链路可行，但部分原始 bbox 会映射到图片外，当前先记录到 `errors.jsonl` 并跳过。
- 384 crop 比原始 crop 更省磁盘和 I/O，适合作为第一版监督分类 baseline 的输入。

### 下一步

- 等待全量 crop 完成后补 `summary.jsonl` 统计。
- 基于 `manifest.jsonl` 做类别长尾统计和验证集切分。
- 启动 ResNet/ConvNeXt 监督分类 smoke。

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
