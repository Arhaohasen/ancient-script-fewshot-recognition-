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

**Verdict：Positive**

已跑通 ResNet50 监督分类 baseline。无预训练 5 epoch 验证准确率 5.21%；ImageNet 预训练 5 epoch 验证准确率 32.21%。发现 `ZHFD-...` 占位标签后，重新过滤并训练 filtered 版本，验证准确率 32.18%，作为后续实验和提交推理的第一版本地基线。

### 设置

- 数据：`/root/autodl-tmp/datasets/processed/char_crops_384`
- 模型：ResNet50
- 关键超参：224 输入，batch size 64，AdamW，`none` lr 1e-3，`imagenet` lr 3e-4
- 验证集：`splits/seed42_valid10`

### 结果

- 本地指标：21 个单元测试通过
- 线上分数：无
- split：train 41,244，valid 5,184，valid 覆盖 1,773 类
- no-pretrain：best epoch 5，valid acc 5.21%
- imagenet：best epoch 5，train acc 46.18%，valid acc 32.21%
- imagenet filtered：train 40,558，valid 5,109，2,999 类，valid acc 32.18%
- run.sh smoke：8 张样本可生成 UTF-8 `prediction.json`，占位标签预测数 0

### 解释

- 单样本类别只进训练集，避免验证集中出现训练完全没见过的类别。
- ImageNet 预训练对古文字 crop 仍然非常有帮助，说明底层纹理/边缘特征迁移有效。
- `ZHFD-...` 等 ASCII 占位标签会污染提交预测，已在 split 阶段过滤。
- 当前验证集来自同一批 crop 的类内切分，不代表线上跨域泛化能力，只能作为内部迭代对照。

### 下一步

- 补推理与 `prediction.json` 生成脚本。
- 确认赛题测试集/提交格式。
- 基于 ImageNet baseline 尝试 ConvNeXt/更强增强/类别重采样。

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
