# 04-code

这里放比赛代码。当前已经有数据裁剪、统计、切分和第一版监督分类训练入口。

---

## 预期模块

```text
04-code/
├── ancient_ocr/
│   ├── data/       # XML 解析、crop 构建、manifest 统计、split
│   └── training/   # 监督分类训练入口
└── tests/          # 单元测试
```

---

## 第一版代码目标

第一阶段代码目标：

1. 能读取训练图片和标签；
2. 能构造训练/验证切分；
3. 能训练一个最小模型；
4. 能跑验证指标；
5. 能对测试集生成提交文件。

---

## 暂定技术栈

- Python
- PyTorch
- torchvision
- pandas / numpy
- PIL

---

## 代码原则

- 数据路径通过配置传入，不写死本机绝对路径；
- 随机种子可固定；
- 每个实验能追溯到配置和代码版本；
- 大文件输出默认写到 `03-实验/exp/expNNN_xxx/` 或 `05-submission/submissions/`。
