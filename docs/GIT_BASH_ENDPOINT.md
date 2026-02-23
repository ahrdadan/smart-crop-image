# Panduan Lengkap Git Bash (Async Queue + Job Result)

Dokumen ini khusus workflow **Git Bash di Windows** untuk:

1. Build image Docker.
2. Menjalankan service API.
3. Submit request ke queue (`POST /thumbnail`).
4. Cek status queue (`GET /job/{id}`).
5. Ambil hasil gambar (`GET /job/{id}/image`).

## 0) Prasyarat

- Docker Desktop aktif.
- Git Bash.

Validasi:

```bash
docker --version
docker info > /dev/null && echo "Docker OK"
```

## 1) Build Docker Image

Jalankan dari root project:

```bash
docker build -t smart-crop-image:latest .
```

## 2) Siapkan Folder Input/Output/Job

```bash
mkdir -p ./docker-data/input ./docker-data/output ./docker-data/jobs ./docker-data/request
cp ./sample/*.webp ./docker-data/input/
```

`jobs` dipakai server untuk menyimpan payload + hasil sementara (akan dibersihkan otomatis setelah TTL).

## 3) Jalankan Container API

Pakai `MSYS_NO_PATHCONV=1` supaya path volume Git Bash tidak diubah.

```bash
MSYS_NO_PATHCONV=1 docker run --rm -d \
  --name smart-crop-api \
  -p 8080:8080 \
  -e PORT=8080 \
  -e JOB_TTL_HOURS=2 \
  -e JOB_QUEUE_CAPACITY=1000 \
  -e THUMBNAIL_PAIR_WIDTH=1200 \
  -v "$(pwd)/docker-data/input:/data/input" \
  -v "$(pwd)/docker-data/output:/data/output" \
  -v "$(pwd)/docker-data/jobs:/data/jobs" \
  smart-crop-image:latest
```

Cek:

```bash
docker ps --filter name=smart-crop-api
docker logs smart-crop-api
curl -sS "http://127.0.0.1:8080/health"
```

## 4) Buat Payload Submit Job

`output_path` dari client tidak diperlukan lagi. Server menentukan sendiri output job.

```bash
cat > ./docker-data/request/payload.json <<'JSON'
{
  "image_paths": [
    "/data/input/01.webp",
    "/data/input/02.webp",
    "/data/input/03.webp",
    "/data/input/04.webp"
  ],
  "return_candidates": true,
  "webhook_url": "https://webhook.site/your-id"
}
JSON
```

Jika tidak butuh webhook, hapus field `webhook_url`.

Catatan:

- `image_paths`/`image_path` dipakai jika server bisa akses path tersebut.
- Untuk server deploy (remote) yang tidak bisa baca file lokal kamu, gunakan `image_files` (Base64) atau pakai script Node dengan `--embed-local-images true`.

## 5) Submit ke Queue (`POST /thumbnail`)

```bash
curl -sS -X POST "http://127.0.0.1:8080/thumbnail" \
  -H "Content-Type: application/json" \
  --data-binary @./docker-data/request/payload.json \
  | tee ./docker-data/output/enqueue_response.json
```

Contoh response:

```json
{
  "job_id": "9e79cdb54cb7d537ad602952",
  "status": "queued",
  "queue_position": 1,
  "pending_jobs": 1,
  "job_url": "http://127.0.0.1:8080/job/9e79cdb54cb7d537ad602952",
  "image_url": "http://127.0.0.1:8080/job/9e79cdb54cb7d537ad602952/image",
  "created_at": "2026-02-23T10:00:00Z",
  "expires_in_seconds": 7200,
  "webhook_enabled": true
}
```

Ambil `job_id` dari file response:

```bash
JOB_ID=$(sed -n 's/.*"job_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' ./docker-data/output/enqueue_response.json | head -n1)
echo "$JOB_ID"
```

## 6) Poll Status Queue (`GET /job/{id}`)

```bash
curl -sS "http://127.0.0.1:8080/job/$JOB_ID" | tee ./docker-data/output/job_status.json
```

Cek sampai `status` menjadi `done` atau `failed`:

```bash
while true; do
  STATUS=$(curl -sS "http://127.0.0.1:8080/job/$JOB_ID" | tee ./docker-data/output/job_status.json | sed -n 's/.*"status"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1)
  echo "status=$STATUS"
  if [[ "$STATUS" == "done" || "$STATUS" == "failed" ]]; then
    break
  fi
  sleep 2
done
```

Status queue:

- `queued`: masih antri.
- `processing`: sedang diproses worker.
- `done`: selesai, gambar siap diakses.
- `failed`: gagal diproses.

Di response status job, field `webhook` berisi progres delivery callback (attempt, delivered, last_error).

## 7) Ambil Hasil Gambar (`GET /job/{id}/image`)

### Default JPG

```bash
curl -sS "http://127.0.0.1:8080/job/$JOB_ID/image" \
  -o ./docker-data/output/result-default.jpg
```

### Konversi AVIF

```bash
curl -sS "http://127.0.0.1:8080/job/$JOB_ID/image?format=avif" \
  -o ./docker-data/output/result.avif
```

### Resize by Width (height otomatis ikut aspect ratio)

```bash
curl -sS "http://127.0.0.1:8080/job/$JOB_ID/image?width=640" \
  -o ./docker-data/output/result-w640.jpg
```

### Quality 0-100 (JPG atau AVIF)

```bash
curl -sS "http://127.0.0.1:8080/job/$JOB_ID/image?format=jpg&quality=80" \
  -o ./docker-data/output/result-q80.jpg

curl -sS "http://127.0.0.1:8080/job/$JOB_ID/image?format=avif&quality=60" \
  -o ./docker-data/output/result-q60.avif
```

## 8) TTL Cleanup (2 Jam Default)

- Setelah job selesai, payload (`request.json`) dan output image job akan dihapus otomatis setelah `2 jam` (default).
- Saat sudah dibersihkan, `GET /job/{id}` akan return `404 job not found or expired`.

## 9) Contoh Single Image

```bash
cat > ./docker-data/request/payload-single.json <<'JSON'
{
  "image_path": "/data/input/01.webp",
  "return_candidates": true
}
JSON
```

```bash
curl -sS -X POST "http://127.0.0.1:8080/thumbnail" \
  -H "Content-Type: application/json" \
  --data-binary @./docker-data/request/payload-single.json
```

## 10) Troubleshooting

`queue is full, please retry later`

- Naikkan `JOB_QUEUE_CAPACITY`.

`job is not completed yet`

- Tunggu status jadi `done` lalu akses endpoint `/image`.

`job not found or expired`

- Job sudah lewat TTL cleanup atau `job_id` salah.

`image convert worker not found`

- Pastikan `image_convert.py` tersedia dan env `IMAGE_CONVERT_WORKER_PATH` benar.

`format must be one of: jpg, avif`

- Query `format` hanya menerima `jpg|avif`.

`quality must be between 0 and 100`

- Query `quality` wajib rentang `0..100`.

`webhook_url must use http or https`

- Pastikan URL webhook diawali `http://` atau `https://`.

`webhook.delivered=false`

- Callback belum terkirim atau gagal; cek field `webhook.last_error` dari endpoint `GET /job/{id}`.

`image_paths, image_path, or image_files is required`

- Kirim salah satu sumber input: `image_paths`, `image_path`, atau `image_files`.
- Untuk server remote, paling aman gunakan `--embed-local-images true` di script Node.

## 11) Stop Container

```bash
docker stop smart-crop-api
```

## 12) Sample Request dari Node.js

Script sample ada di:

- `sample/nodejs/submit_thumbnail_job.mjs`
- `sample/nodejs/payload.local.json`
- `sample/nodejs/payload.docker.json`
- `sample/nodejs/payload.webhook.local.json`

Prasyarat:

- Node.js 18+ (karena memakai `fetch` bawaan).

### Jalankan untuk server local (`go run .`)

```bash
node sample/nodejs/submit_thumbnail_job.mjs \
  --api http://127.0.0.1:8080 \
  --payload sample/nodejs/payload.local.json \
  --output sample/nodejs/out/result-local.jpg \
  --format jpg \
  --quality 90
```

### Jalankan untuk server Docker (path `/data/input/...`)

```bash
node sample/nodejs/submit_thumbnail_job.mjs \
  --api http://127.0.0.1:8080 \
  --payload sample/nodejs/payload.docker.json \
  --output sample/nodejs/out/result-docker.avif \
  --format avif \
  --width 640 \
  --quality 65
```

### Jalankan dengan webhook callback

```bash
node sample/nodejs/submit_thumbnail_job.mjs \
  --api http://127.0.0.1:8080 \
  --payload sample/nodejs/payload.webhook.local.json \
  --output sample/nodejs/out/result-webhook.jpg \
  --format jpg \
  --quality 90
```

### Jalankan ke server deploy (remote) dengan file lokal

```bash
node sample/nodejs/submit_thumbnail_job.mjs \
  --api https://your-deployed-domain \
  --payload sample/nodejs/payload.local.json \
  --output sample/nodejs/out/result-remote.jpg \
  --embed-local-images true \
  --format jpg \
  --quality 90
```

Output script:

- File gambar hasil (sesuai `--output`)
- File status JSON di `<output>.status.json`

Opsi script:

- `--api` default `http://127.0.0.1:8080`
- `--payload` default `sample/nodejs/payload.local.json`
- `--output` default `sample/nodejs/out/result.jpg`
- `--format` `jpg|jpeg|avif` (default `jpg`)
- `--width` default `0` (tanpa resize)
- `--quality` `0..100` (default `95`)
- `--embed-local-images` `true|false` (default `false`)
- `--poll-interval-ms` default `2000`
- `--timeout-ms` default `600000`
