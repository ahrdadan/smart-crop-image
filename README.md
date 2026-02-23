# Smart Crop Image (Single Endpoint + Python Pair Composer)

Project ini memakai arsitektur **Go API + Python worker** dengan **satu endpoint** saja:

- `POST /thumbnail`

Endpoint tersebut akan:
- memilih 2 halaman terbaik dari input chapter,
- menggabungkannya menjadi thumbnail 16:9 di Python (`smart_thumb.py`),
- mengembalikan metadata hasilnya.

## Struktur

- `main.go`  
  API Go single endpoint (`/thumbnail`) yang memanggil `smart_thumb.py`.
- `smart_thumb.py`  
  Worker Python untuk ranking halaman + merge 2 gambar jadi 16:9.
- `scripts/start_background.sh`  
  Script setup dependency, build, start background, dan sample test.

## Jalur Proses

1. Client kirim request ke `POST /thumbnail`.
2. Go menyiapkan payload worker (`image_paths`, `output_path`, parameter pair).
3. Go memanggil `smart_thumb.py`.
4. Python memilih 2 gambar terbaik dan menghasilkan file output merge 16:9.
5. Go mengembalikan response JSON.

## Menjalankan API

```bash
go run .
```

Server listen default di `:8080`.

Atau pakai script:

```bash
./scripts/start_background.sh
```

## Request API

Endpoint:

```http
POST /thumbnail
Content-Type: application/json
```

Contoh payload chapter:

```json
{
  "image_paths": [
    "D:/data/ch001/page-001.jpg",
    "D:/data/ch001/page-002.jpg",
    "D:/data/ch001/page-003.jpg"
  ],
  "output_path": "D:/data/ch001/ch001-thumb.jpg",
  "return_candidates": true
}
```

Payload juga bisa pakai `image_path` tunggal.  
Jika hanya 1 gambar valid, worker akan duplikasi panel agar tetap jadi 16:9 pair output.

## Response Contoh

```json
{
  "crop_x": 0,
  "crop_y": 0,
  "crop_width": 1200,
  "crop_height": 675,
  "method": "pair-smart-thumb",
  "confidence": 1,
  "applied": true,
  "output_path": "D:/data/ch001/ch001-thumb.jpg",
  "composition_mode": "pair-smart-thumb",
  "composed_from": [
    "D:/data/ch001/page-003.jpg",
    "D:/data/ch001/page-007.jpg"
  ],
  "selected_page_index": 2,
  "selected_image_path": "D:/data/ch001/page-003.jpg",
  "selected_score": 1.9821
}
```

Jika worker gagal, API tetap merespons dengan `crop_error`.

## Sample Test

Jalankan:

```bash
SAMPLE_COMPOSE_MODE=pair ./scripts/start_background.sh
```

Output sample:
- `sample/out/thumbnail_response.json`
- `sample/out/thumbnail_payload.json`
- `sample/out/chapter-thumbnail.jpg`
- `logs/start_background_*.log`

## Environment Variables

- `PORT`  
  Port API Go. Default `8080`.
- `PYTHON_BIN`  
  Command Python custom, contoh: `python` atau `py -3`.
- `SMART_THUMB_WORKER_PATH`  
  Path worker pair Python. Default `smart_thumb.py`.
- `THUMBNAIL_YOLO_MODEL`  
  Model YOLO untuk worker pair. Default di worker: `yoloe-26s-seg.pt`.
- `THUMBNAIL_YOLO_DEVICE`  
  Device inference. Default `cpu`.
- `THUMBNAIL_PAIR_SKIP_EDGES`  
  Parameter `skip_edges`. Default `2`.
- `THUMBNAIL_PAIR_GAP`  
  Jarak antar panel. Default `5`.
- `THUMBNAIL_PAIR_WIDTH`  
  Lebar output. Default `1200`.

## Dependency Runtime

- Python packages:
  - `numpy`
  - `Pillow`
  - `opencv-contrib-python-headless`
  - `ultralytics`

## Build Check

```bash
gofmt -w main.go
go build ./...
```
