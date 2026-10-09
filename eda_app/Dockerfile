FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EDA_DEBUG=0 \
    EDA_UPLOAD_DIR=/data/uploads \
    EDA_REPORT_DIR=/data/reports \
    EDA_METADATA_FILE=/data/metadata.json \
    EDA_SECRET_KEY_FILE=/data/.secret_key \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --system --uid 10001 --no-create-home appuser \
    && mkdir -p /data/uploads /data/reports /tmp/matplotlib \
    && chown -R appuser:appuser /data /tmp/matplotlib

USER appuser

EXPOSE 8501

CMD ["gunicorn", "--bind", "0.0.0.0:8501", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
