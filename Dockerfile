FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLLP_ADDRESS=localhost:8440 \
    PAGER_ADDRESS=localhost:8441 \
    PROMETHEUS_PORT=8000 \
    MODEL_PATH=/app/model/model.onnx \
    THRESHOLD_PATH=/app/model/threshold.txt \
    HISTORY_PATH=/data/history.csv \
    PERSISTENCE_PATH=/state/patients.pkl \
    JOURNAL_PATH=/state/journal.jsonl \
    SAVE_INTERVAL=100 \
    LOG_LEVEL=INFO \
    RECONNECT=true

WORKDIR /app

COPY requirements.txt .
RUN mkdir -p /state && chmod 777 /state && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app/
COPY model/ /app/model/

EXPOSE 8000
CMD ["python", "-m", "app.main"]