# Smart Crop Image (Async Queue + Job Endpoint)

Service ini memakai arsitektur **Go API + Python worker** dengan flow async:

1. Client kirim `POST /thumbnail`.
2. Server menaruh request ke queue.
3. Client polling `GET /job/{id}`.
4. Saat selesai, hasil gambar diakses lewat `GET /job/{id}/image`.

## Endpoint

- `POST /thumbnail`  
  Submit job baru (async queue).
- `GET /job/{id}`  
  Cek status queue/progress/result.
- `GET /job/{id}/image?format=jpg|avif&width=...&quality=...`  
  Ambil hasil gambar dengan opsi konversi.
- `webhook_url` (opsional di payload `POST /thumbnail`)  
  Jika diisi, server akan POST callback saat job selesai (`done` atau `failed`).

## Fitur Utama

- Single worker processing (job diproses berurutan).
- Queue position di response.
- TTL job artifact: payload + output gambar dihapus otomatis setelah **2 jam** (default).
- Output path ditentukan server per job, bukan dari client.
- Callback webhook async dengan retry.

## Docker Quick Start

```bash
docker build -t smart-crop-image:latest .
```

```bash
MSYS_NO_PATHCONV=1 docker run --rm -d \
  --name smart-crop-api \
  -p 8080:8080 \
  -e PORT=8080 \
  -v "$(pwd)/docker-data/input:/data/input" \
  -v "$(pwd)/docker-data/output:/data/output" \
  -v "$(pwd)/docker-data/jobs:/data/jobs" \
  smart-crop-image:latest
```

Panduan Git Bash lengkap:

- `docs/GIT_BASH_ENDPOINT.md`

Sample client Node.js:

- `sample/nodejs/submit_thumbnail_job.mjs`

## Environment Variables

- `PORT` (default `8080`)
- `PYTHON_BIN` (default auto detect: `python`, `python3`, `py -3`)
- `SMART_THUMB_WORKER_PATH` (default `smart_thumb.py`)
- `IMAGE_CONVERT_WORKER_PATH` (default `image_convert.py`)
- `JOB_STORAGE_DIR` (default `job-data`)
- `JOB_QUEUE_CAPACITY` (default `1000`)
- `JOB_TTL_HOURS` (default `2`)
- `JOB_PROCESS_TIMEOUT_SECONDS` (default `1800`)
- `WEBHOOK_TIMEOUT_SECONDS` (default `15`)
- `WEBHOOK_RETRIES` (default `3`)
- `WEBHOOK_BACKOFF_MS` (default `2000`)
- `THUMBNAIL_YOLO_MODEL` (default worker: `yoloe-26s-seg.pt`)
- `THUMBNAIL_YOLO_DEVICE` (default `cpu`)
- `THUMBNAIL_PAIR_SKIP_EDGES` (default `2`)
- `THUMBNAIL_PAIR_GAP` (default `5`)
- `THUMBNAIL_PAIR_WIDTH` (default `1200`)

## Dependency Runtime

- `numpy`
- `Pillow`
- `opencv-contrib-python-headless`
- `ultralytics`
- `pillow-avif-plugin`

## Build Check

```bash
gofmt -w main.go
go build ./...
```

## Webhook Callback

Jika request `POST /thumbnail` mengirim `webhook_url`, saat job selesai server mengirim:

- Method: `POST`
- Header: `Content-Type: application/json`
- Body berisi metadata job (`job_id`, `status`, `job_url`, `image_url`, `result`, `error`, timestamp).

Contoh field payload callback:

- `event`: `thumbnail.job.completed`
- `status`: `done` atau `failed`
- `result.output_path`: URL endpoint image (bukan path file internal)
