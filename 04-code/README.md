# 04-code

这里放比赛代码。当前先不急着实现，等数据格式确认后再创建具体脚本。

---

## 预期模块

```text
04-code/
├── data/       # Dataset、数据清洗、切分、统计
├── models/     # 模型定义
├── training/   # 训练脚本、loss、scheduler
├── inference/  # 推理与提交文件生成
└── utils/      # 通用工具
```

---

## 第一版代码目标

`exp001_baseline_pipeline_smoke` 只要求：

1. 能读取训练图片和标签；
2. 能构造训练/验证切分；
3. 能训练一个最小模型；
4. 能跑验证指标；
5. 能对测试集生成提交文件。

---

## 暂定技术栈

- Python
- PyTorch
- torchvision / timm（待确认是否安装和允许）
- pandas / numpy
- PIL / OpenCV

---

## 代码原则

- 数据路径通过配置传入，不写死本机绝对路径；
- 随机种子可固定；
- 每个实验能追溯到配置和代码版本；
- 大文件输出默认写到 `03-实验/exp/expNNN_xxx/` 或 `05-submission/submissions/`。
