#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
SAMPLE_DIR="${SAMPLE_DIR:-$ROOT_DIR/sample}"
OUT_DIR="${OUT_DIR:-$SAMPLE_DIR/out}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
RESP_FILE="${RESP_FILE:-$OUT_DIR/thumbnail_response.json}"
PAYLOAD_FILE="${PAYLOAD_FILE:-$OUT_DIR/thumbnail_payload.json}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-$OUT_DIR/chapter-thumbnail.jpg}"
TEST_LOG="${TEST_LOG:-$LOG_DIR/test_sample_thumbnail_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
exec > >(tee -a "$TEST_LOG") 2>&1

source "$ROOT_DIR/scripts/common.sh"

OS_NAME="$(detect_os)"
ARCH_NAME="$(detect_arch)"
echo "Detected OS: $OS_NAME | ARCH: $ARCH_NAME"
if [[ "$OS_NAME" == "unsupported" || "$ARCH_NAME" == "unsupported" ]]; then
  echo "Unsupported OS/architecture combination."
  exit 1
fi

echo "Checking dependencies and auto-installing when missing..."
ensure_curl
ensure_python3
echo "Using python: $(command -v python3)"
echo "Using curl: $(command -v curl)"

if ! curl -fsS "http://$HOST:$PORT/healthz" >/dev/null 2>&1; then
  echo "Server is not reachable at http://$HOST:$PORT"
  echo "Run: ./scripts/start_background.sh"
  echo "Test log: $TEST_LOG"
  exit 1
fi

mapfile -t IMAGES < <(find "$SAMPLE_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | sort)

if [[ ${#IMAGES[@]} -eq 0 ]]; then
  echo "No sample images found in $SAMPLE_DIR"
  echo "Test log: $TEST_LOG"
  exit 1
fi

python3 - "$PAYLOAD_FILE" "$OUTPUT_IMAGE" "${IMAGES[@]}" <<'PY'
import json
import sys

payload_path = sys.argv[1]
output_image = sys.argv[2]
image_paths = sys.argv[3:]

payload = {
    "image_paths": image_paths,
    "preferred_ratio": "16:9",
    "max_analysis_size": 512,
    "apply_crop": True,
    "output_path": output_image,
    "quality": 85,
    "return_candidates": True,
}

with open(payload_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
PY

curl -fsS -X POST "http://$HOST:$PORT/thumbnail" \
  -H "Content-Type: application/json" \
  --data @"$PAYLOAD_FILE" \
  >"$RESP_FILE"

python3 - "$RESP_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)

print("Selected page index:", data.get("selected_page_index"))
print("Selected image path:", data.get("selected_image_path"))
print("Method:", data.get("method"))
print("Confidence:", data.get("confidence"))
print("Applied:", data.get("applied"))
print("Output path:", data.get("output_path"))
if data.get("crop_error"):
    print("Crop error:", data.get("crop_error"))
if data.get("worker_warning"):
    print("Worker warning:", data.get("worker_warning"))
PY

if [[ -f "$OUTPUT_IMAGE" ]]; then
  echo "Thumbnail generated: $OUTPUT_IMAGE"
  echo "Response JSON: $RESP_FILE"
  echo "Test log: $TEST_LOG"
else
  echo "Thumbnail image not found. Check response: $RESP_FILE"
  echo "Test log: $TEST_LOG"
  exit 1
fi
