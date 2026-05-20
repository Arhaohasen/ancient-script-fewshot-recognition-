# 03-实验/exp

每个子目录对应一个实验。

---

## 命名约定

```text
expNNN_{stage}_{method}_{shortname}/
```

示例：

```text
exp001_baseline_pipeline_smoke/
exp002_supervised_resnet50_strongaug/
exp003_fewshot_proto_vit/
```

---

## 每个实验目录建议内容

```text
expNNN_xxx/
├── EXP.md
├── configs/
│   └── override.yaml
├── logs/
├── ckpt/
├── artifacts/
└── notes.md
```

`EXP.md` 和小型 `artifacts/` 可保留；大日志、checkpoint、原始提交压缩包不进 git。

---

## 实验推进顺序

1. Smoke：小样本、少步数，只验证流程不报错。
2. Pilot：小规模训练，观察 loss 是否下降。
3. Full：完整训练，生成可提交结果。
4. Submission：提交线上，记录反馈。
