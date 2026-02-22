# Smart Crop Image (Manga/Manhwa Thumbnail 16:9)

Project ini berisi arsitektur **Go + Python worker** untuk generate area crop thumbnail 16:9 secara otomatis, dengan prioritas karakter/wajah/area kontras tinggi dan fallback deterministik.

## Struktur

- `main.go`  
  API Go (`POST /thumbnail`) yang memanggil worker Python, mengembalikan koordinat crop, dan opsional mengeksekusi crop final pakai `libvips`.
- `thumbnail_worker.py`  
  AI worker pipeline: `saliency -> face -> fallback`.
- `go.mod`  
  Modul Go.

## Pipeline (Python Worker)

Urutan sesuai requirement:

1. Preprocess: load image, resize max-side 512 (analysis only), grayscale untuk analisis.
2. Saliency (primary): SpectralResidual/OpenCV, top 30% saliency, expand ke 16:9, score confidence.
3. Face detection (secondary): dipakai jika confidence saliency rendah. Prioritas model ONNX (`onnxruntime`, CPU) lalu fallback ke cascade.
4. Fallback deterministic (mandatory): top-biased crop (~18%), full width jika memungkinkan.
5. Output JSON wajib (tidak pernah `null`, tidak di luar bounds).

Output:

```json
{
  "crop_x": 0,
  "crop_y": 120,
  "crop_width": 1080,
  "crop_height": 608,
  "method": "saliency",
  "confidence": 0.81
}
```

## Menjalankan API Go

```bash
go run .
```

Default listen di `:8080`.

Health check:

```bash
curl http://localhost:8080/healthz
```

Atau pakai script background (install dependency + build + start):

```bash
./scripts/start_background.sh
```

`start_background.sh` akan:
- deteksi OS + arsitektur
- auto-install dependency yang belum ada (`python3`, `go`, `curl`, dan mencoba `vips`)
- membuat/repair virtualenv (dengan fallback jika `python3 -m venv` error)
- menjalankan API di background (`nohup`)
- otomatis test endpoint `/thumbnail` menggunakan semua gambar di folder `sample/`
- menyimpan semua output (setup + server + test) ke **satu file log**: `logs/start_background_*.log`

## Request API

Endpoint:

```http
POST /thumbnail
Content-Type: application/json
```

Contoh payload (single image):

```json
{
  "image_path": "D:/data/page-001.jpg",
  "image_width": 1080,
  "image_height": 1920,
  "preferred_ratio": "16:9",
  "max_analysis_size": 512,
  "apply_crop": true,
  "output_path": "D:/data/page-001-thumb.jpg",
  "quality": 85
}
```

Contoh payload (chapter banyak halaman):

```json
{
  "image_paths": [
    "D:/data/ch001/page-001.jpg",
    "D:/data/ch001/page-002.jpg",
    "D:/data/ch001/page-003.jpg"
  ],
  "preferred_ratio": "16:9",
  "max_analysis_size": 512,
  "apply_crop": true,
  "output_path": "D:/data/ch001/ch001-thumb.jpg",
  "quality": 85,
  "return_candidates": true
}
```

Contoh `curl`:

```bash
curl -X POST http://localhost:8080/thumbnail \
  -H "Content-Type: application/json" \
  -d "{\"image_path\":\"D:/data/page-001.jpg\",\"image_width\":1080,\"image_height\":1920,\"preferred_ratio\":\"16:9\",\"max_analysis_size\":512,\"apply_crop\":true,\"output_path\":\"D:/data/page-001-thumb.jpg\",\"quality\":85}"
```

Contoh response (chapter mode):

```json
{
  "crop_x": 120,
  "crop_y": 410,
  "crop_width": 1080,
  "crop_height": 608,
  "method": "saliency",
  "confidence": 0.81,
  "applied": true,
  "output_path": "D:/data/ch001/ch001-thumb.jpg",
  "selected_page_index": 2,
  "selected_image_path": "D:/data/ch001/page-003.jpg",
  "selected_score": 0.8575,
  "candidates": [
    {
      "page_index": 0,
      "image_path": "D:/data/ch001/page-001.jpg",
      "crop_x": 0,
      "crop_y": 334,
      "crop_width": 1080,
      "crop_height": 608,
      "method": "fallback",
      "confidence": 0,
      "score": 0.1375
    }
  ]
}
```

Jika crop file gagal (mis. `vips` belum terpasang), API tetap mengembalikan koordinat + field `crop_error`.

## Test Dengan Folder `/sample`

Test sample sekarang otomatis dijalankan oleh:

```bash
./scripts/start_background.sh
```

Taruh file gambar chapter (jpg/png/webp) di folder `sample/` sebelum menjalankan script.  
Kalau ingin start API tanpa test sample:

```bash
RUN_SAMPLE_TEST=0 ./scripts/start_background.sh
```

Output yang dihasilkan:
- response JSON di `sample/out/thumbnail_response.json`
- payload yang dipakai di `sample/out/thumbnail_payload.json`
- thumbnail hasil crop di `sample/out/chapter-thumbnail.jpg`
- satu file log terpadu di `logs/start_background_*.log`

## Environment Variables

- `PORT`  
  Port API Go. Default `8080`.
- `RUN_SAMPLE_TEST`  
  `1` (default) untuk auto-test sample, `0` untuk skip test.
- `SAMPLE_APPLY_CROP`  
  `1` (default) untuk test dengan crop fisik jika `vips` tersedia, `0` untuk test koordinat saja.
- `AUTO_PORT_FALLBACK`  
  `1` (default) jika port dipakai proses lain maka otomatis pindah ke port kosong berikutnya.
- `LOG_FILE`  
  Path log gabungan. Default `logs/start_background_<timestamp>.log`.
- `THUMBNAIL_WORKER_PATH`  
  Path ke file worker Python. Default `thumbnail_worker.py`.
- `PYTHON_BIN`  
  Command Python custom, contoh: `python` atau `py -3`.
- `ANIME_FACE_CASCADE_PATH`  
  Path cascade XML anime face (opsional, dibaca oleh worker Python).
- `ANIME_FACE_ONNX_PATH`  
  Path model ONNX face/anime-face detector (opsional, prioritas utama untuk AI-assisted detection).
- `apply_crop` / `output_path` / `quality` (di body request)  
  Mengaktifkan crop fisik dengan `vips crop`. Jika `output_path` kosong dan `apply_crop=true`, nama default: `<nama_file>_thumb.<ext>`.
- `image_paths` (di body request)  
  Untuk mode chapter. API akan evaluasi tiap halaman, lalu pilih halaman terbaik secara deterministik.
- `return_candidates` (di body request)  
  Jika `true`, response menyertakan daftar skor tiap halaman (`candidates`).

## Dependency Runtime

- Python + package:
  - `numpy`
  - `Pillow`
  - `opencv-contrib-python-headless`
  - `onnxruntime`
- `libvips` CLI (`vips`) harus tersedia di PATH jika ingin `apply_crop=true`.

## Catatan Integrasi

- Jika worker Python gagal dijalankan, Go otomatis mengembalikan fallback crop deterministik.
- Jika crop fisik gagal, koordinat tetap dikembalikan dan error ada di field `crop_error`.
- Untuk chapter, output crop final selalu diambil dari halaman terpilih (`selected_image_path`).

## Build Check

Sudah diverifikasi:

```bash
gofmt -w main.go
go build ./...
```
