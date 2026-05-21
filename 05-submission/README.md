# 05-submission

这里放提交相关文件和记录。真正提交前先看 `00-赛题资料/提交流程.md`。

当前赛方要求是镜像提交：

- `/app/run.sh`：运行入口；
- `/saisdata`：输入数据目录；
- `/saisresult/prediction.json`：固定输出文件；
- `models/best.pt`：构建镜像前复制到仓库根目录下的模型权重，不进 git；
- 截止时间：2026-05-27 14:00:00。

---

## 目录

```text
05-submission/
├── README.md
└── submissions/
```

---

## 镜像构建 SOP

当前 Dockerfile 使用 PyTorch CUDA 12.4 runtime 基础镜像，适配赛方 V100 + CUDA 12.4 环境。

在 AutoDL 仓库根目录执行：

```bash
mkdir -p models
cp /root/autodl-tmp/runs/exp002_supervised_resnet50_imagenet_filtered_full/best.pt models/best.pt
docker build -t ancient-script-submit:exp002 .
```

本地或 AutoDL smoke：

```bash
mkdir -p /tmp/saisresult_docker
docker run --rm --gpus all \
  -v /path/to/saisdata:/saisdata:ro \
  -v /tmp/saisresult_docker:/saisresult \
  ancient-script-submit:exp002
```

验证输出：

```bash
python -m json.tool /tmp/saisresult_docker/prediction.json >/dev/null
```

注意：`models/best.pt` 会进入 Docker 镜像，但不会提交到 git。

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
