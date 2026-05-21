FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/04-code

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY 04-code/ /app/04-code/
COPY run.sh /app/run.sh
COPY models/best.pt /app/best.pt

RUN chmod +x /app/run.sh && mkdir -p /saisresult

ENTRYPOINT ["bash", "/app/run.sh"]
