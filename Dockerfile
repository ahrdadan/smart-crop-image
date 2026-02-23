FROM golang:1.22-bookworm AS builder

WORKDIR /src

COPY go.mod ./
COPY main.go ./
RUN go mod download
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o /out/smart-crop-image .

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PYTHON_BIN=python3 \
    SMART_THUMB_WORKER_PATH=/app/smart_thumb.py \
    IMAGE_CONVERT_WORKER_PATH=/app/image_convert.py \
    JOB_STORAGE_DIR=/data/jobs \
    JOB_TTL_HOURS=2 \
    THUMBNAIL_YOLO_DEVICE=cpu \
    CUDA_VISIBLE_DEVICES=-1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY smart_thumb.py ./
COPY image_convert.py ./
COPY --from=builder /out/smart-crop-image /app/smart-crop-image

RUN mkdir -p /data/input /data/output /data/jobs

EXPOSE 8080

ENTRYPOINT ["/app/smart-crop-image"]
