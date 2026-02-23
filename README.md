---
title: Smart Thumb Crop
emoji: 📊
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
---

# Smart Crop Image (Async Queue + Job Endpoint)

Service ini memakai arsitektur **Go API + Python worker** dengan flow async:

1. Client kirim `POST /thumbnail`.
2. Server menaruh request ke queue.
3. Client polling `GET /job/{id}`.
4. Saat selesai, hasil gambar diakses lewat `GET /job/{id}/image`.

## Endpoint

- `GET /health` atau `GET /healthz`  
  Health check service + metrik queue.
- `POST /thumbnail`  
  Submit job baru (async queue).
- `GET /job/{id}`  
  Cek status queue/progress/result.
- `GET /job/{id}/image?format=jpg|avif&width=...&quality=...`  
  Ambil hasil gambar dengan opsi konversi.
- `image_files` (opsional di payload `POST /thumbnail`)  
  Upload gambar langsung via Base64 untuk server remote (tanpa shared filesystem).
- `webhook_url` (opsional di payload `POST /thumbnail`)  
  Jika diisi, server akan POST callback saat job selesai (`done` atau `failed`).

## Fitur Utama

- Single worker processing (job diproses berurutan).
- Queue position di response.
- TTL job artifact: payload + output gambar dihapus otomatis setelah **2 jam** (default).
- Output path ditentukan server per job, bukan dari client.
- Callback webhook async dengan retry.
- Dukungan `image_files` Base64 untuk kirim file dari client ke server deploy.
- Auto fallback worker ringan saat worker utama gagal (contoh: `signal: killed` / OOM).

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
- mode remote: gunakan flag `--embed-local-images true` agar `image_paths` lokal dikirim sebagai `image_files`.

## Cara Menjalankan Sample Node.js

Pastikan API sudah jalan di `:8080` (lokal atau URL deploy).

Jalankan dari root project:

```bash
node sample/nodejs/submit_thumbnail_job.mjs \
  --api http://127.0.0.1:8080 \
  --payload sample/nodejs/payload.local.json \
  --output sample/nodejs/out/result-local.jpg \
  --format jpg \
  --quality 90
```

Jalankan dari folder `sample/nodejs`:

```bash
cd sample/nodejs
node submit_thumbnail_job.mjs \
  --api http://127.0.0.1:8080 \
  --payload ./payload.local.json \
  --output ./out/result-local.jpg \
  --format jpg \
  --quality 90
```

Jalankan ke server deploy (remote) dari folder `sample/nodejs`:

```bash
cd sample/nodejs
node submit_thumbnail_job.mjs \
  --api https://your-deployed-domain \
  --payload ./payload.local.json \
  --output ./out/result-remote.jpg \
  --embed-local-images true \
  --format jpg \
  --quality 90
```

## Environment Variables

- `PORT` (default `8080`)
- `PYTHON_BIN` (default auto detect: `python`, `python3`, `py -3`)
- `SMART_THUMB_WORKER_PATH` (default `smart_thumb.py`)
- `SMART_THUMB_FALLBACK_WORKER_PATH` (default `fallback_thumb.py`)
- `IMAGE_CONVERT_WORKER_PATH` (default `image_convert.py`)
- `JOB_STORAGE_DIR` (default `job-data`)
- `JOB_QUEUE_CAPACITY` (default `1000`)
- `JOB_TTL_HOURS` (default `2`)
- `JOB_PROCESS_TIMEOUT_SECONDS` (default `1800`)
- `WEBHOOK_TIMEOUT_SECONDS` (default `15`)
- `WEBHOOK_RETRIES` (default `3`)
- `WEBHOOK_BACKOFF_MS` (default `2000`)
- `INLINE_IMAGE_MAX_BYTES` (default `20971520`)
- `INLINE_IMAGE_MAX_COUNT` (default `50`)
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

## Fallback Behavior

Jika worker utama gagal, server otomatis mencoba fallback composer ringan yang tetap menghasilkan:

- 2 panel portrait
- output 16:9
- gap default `5` (mengikuti `THUMBNAIL_PAIR_GAP`)

Penanda fallback di response job (`GET /job/{id}`):

- `result.fallback_used = true`
- `result.worker_warning` berisi alasan fallback
