# 05-submission

这里放提交相关文件和记录。真正提交前先看 `00-赛题资料/提交流程.md`。

当前赛方要求是镜像提交：

- `/app/run.sh`：运行入口；
- `/saisdata`：输入数据目录；
- `/saisresult/prediction.json`：固定输出文件；
- 截止时间：2026-05-27 14:00:00。

---

## 目录

```text
05-submission/
├── README.md
└── submissions/
```

---

## 本地调试文件命名

```text
prediction_YYYYMMDD_HHMM_expNNN_shortname.json
```

镜像内最终文件名必须是：

```text
/saisresult/prediction.json
```

---

## 提交后必须记录

1. 文件路径；
2. 对应实验；
3. 线上分数；
4. 与上一次提交相比的变化；
5. 下一步判断。
