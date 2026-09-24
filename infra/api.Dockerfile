FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src:/app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY api /app/api
COPY runtime /app/runtime
COPY src /app/src
COPY config /app/config
COPY data/datasets /app/data/datasets

EXPOSE 8000
CMD ["python", "-m", "runtime.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
