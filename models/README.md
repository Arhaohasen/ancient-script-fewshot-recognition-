# models

Place the submission checkpoint here before building the Docker image:

```bash
cp /root/autodl-tmp/runs/exp002_supervised_resnet50_imagenet_filtered_full/best.pt models/best.pt
```

`models/best.pt` is required by the Dockerfile but is intentionally ignored by git.
