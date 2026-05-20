# runs.jsonl Schema

`runs.jsonl` 是机器可读实验流水账，每行一个 JSON 对象。它不替代 `JOURNAL.md`，只记录结构化字段。

---

## 字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `run_id` | string | 是 | 实验 ID，如 `exp001` |
| `name` | string | 是 | 实验名称 |
| `started_at` | string | 是 | 开始时间，ISO 或 `YYYY-MM-DD HH:mm` |
| `status` | string | 是 | `planned` / `running` / `done` / `failed` |
| `stage` | string | 是 | `smoke` / `pilot` / `full` / `submission` |
| `data` | object | 是 | 数据版本、切分方式 |
| `model` | object | 是 | 模型和预训练信息 |
| `metrics` | object | 否 | 验证集和线上指标 |
| `artifacts` | object | 否 | checkpoint、提交文件、日志路径 |
| `notes` | string | 否 | 简短备注，详细解释写 `JOURNAL.md` |

---

## 示例

```json
{"run_id":"exp001","name":"baseline_pipeline_smoke","started_at":"2026-05-11 00:30","status":"planned","stage":"smoke","data":{"version":"raw_train_tar","split":"pending"},"model":{"name":"resnet18","pretrained":true},"metrics":{},"artifacts":{},"notes":"先跑通端到端流程"}
```
