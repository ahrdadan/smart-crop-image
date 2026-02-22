#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
VENV_DIR="${VENV_DIR:-.venv}"
LOG_DIR="${LOG_DIR:-logs}"
RUN_DIR="${RUN_DIR:-run}"
BIN_DIR="${BIN_DIR:-bin}"
BIN_PATH="${BIN_PATH:-$BIN_DIR/smart-crop-image}"
PID_FILE="${PID_FILE:-$RUN_DIR/server.pid}"
SAMPLE_DIR="${SAMPLE_DIR:-$ROOT_DIR/sample}"
OUT_DIR="${OUT_DIR:-$SAMPLE_DIR/out}"
RESP_FILE="${RESP_FILE:-$OUT_DIR/thumbnail_response.json}"
PAYLOAD_FILE="${PAYLOAD_FILE:-$OUT_DIR/thumbnail_payload.json}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-$OUT_DIR/chapter-thumbnail.jpg}"
RUN_SAMPLE_TEST="${RUN_SAMPLE_TEST:-1}"
SAMPLE_APPLY_CROP="${SAMPLE_APPLY_CROP:-1}"
AUTO_PORT_FALLBACK="${AUTO_PORT_FALLBACK:-1}"
GO_VERSION="${GO_VERSION:-1.22.12}"
GO_INSTALL_ROOT="${GO_INSTALL_ROOT:-$HOME/.local}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/start_background_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$LOG_DIR" "$RUN_DIR" "$BIN_DIR" "$OUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

detect_os() {
  local kernel
  kernel="$(uname -s | tr '[:upper:]' '[:lower:]')"
  case "$kernel" in
    linux*) echo "linux" ;;
    darwin*) echo "darwin" ;;
    *) echo "unsupported" ;;
  esac
}

detect_arch() {
  local arch
  arch="$(uname -m | tr '[:upper:]' '[:lower:]')"
  case "$arch" in
    x86_64|amd64) echo "amd64" ;;
    aarch64|arm64) echo "arm64" ;;
    armv7l|armv6l) echo "armv6l" ;;
    i386|i686) echo "386" ;;
    *) echo "unsupported" ;;
  esac
}

detect_pkg_manager() {
  if command -v apt-get >/dev/null 2>&1; then echo "apt"; return; fi
  if command -v dnf >/dev/null 2>&1; then echo "dnf"; return; fi
  if command -v yum >/dev/null 2>&1; then echo "yum"; return; fi
  if command -v pacman >/dev/null 2>&1; then echo "pacman"; return; fi
  if command -v zypper >/dev/null 2>&1; then echo "zypper"; return; fi
  if command -v apk >/dev/null 2>&1; then echo "apk"; return; fi
  if command -v brew >/dev/null 2>&1; then echo "brew"; return; fi
  echo "none"
}

run_privileged() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
    return
  fi
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
    return
  fi
  echo "Need root privileges to install packages, but sudo is not available."
  return 1
}

apt_update_once() {
  if [[ "${_APT_UPDATED:-0}" == "1" ]]; then
    return
  fi
  run_privileged apt-get update
  _APT_UPDATED=1
}

install_packages() {
  local manager="$1"
  shift
  local packages=("$@")
  [[ ${#packages[@]} -eq 0 ]] && return

  case "$manager" in
    apt)
      apt_update_once
      run_privileged apt-get install -y "${packages[@]}"
      ;;
    dnf)
      run_privileged dnf install -y "${packages[@]}"
      ;;
    yum)
      run_privileged yum install -y "${packages[@]}"
      ;;
    pacman)
      run_privileged pacman -Sy --noconfirm "${packages[@]}"
      ;;
    zypper)
      run_privileged zypper --non-interactive install "${packages[@]}"
      ;;
    apk)
      run_privileged apk add --no-cache "${packages[@]}"
      ;;
    brew)
      brew install "${packages[@]}"
      ;;
    *)
      echo "No supported package manager found for auto-install."
      return 1
      ;;
  esac
}

ensure_curl() {
  command -v curl >/dev/null 2>&1 && return
  local manager
  manager="$(detect_pkg_manager)"
  case "$manager" in
    apt|dnf|yum|pacman|zypper|apk|brew) install_packages "$manager" curl ;;
    *) echo "curl not found and cannot auto-install."; return 1 ;;
  esac
  command -v curl >/dev/null 2>&1
}

ensure_python3() {
  local manager
  if command -v python3 >/dev/null 2>&1 && python3 -m pip --version >/dev/null 2>&1; then
    return
  fi

  manager="$(detect_pkg_manager)"
  case "$manager" in
    apt)
      install_packages "$manager" python3 python3-pip python3-venv || install_packages "$manager" python3 python3-pip python3-full
      ;;
    dnf|yum)
      install_packages "$manager" python3 python3-pip
      ;;
    pacman)
      install_packages "$manager" python python-pip
      ;;
    zypper)
      install_packages "$manager" python3 python3-pip python3-virtualenv
      ;;
    apk)
      install_packages "$manager" python3 py3-pip py3-virtualenv
      ;;
    brew)
      install_packages "$manager" python
      ;;
    *)
      echo "python3 not found and cannot auto-install."
      return 1
      ;;
  esac
  command -v python3 >/dev/null 2>&1
}

install_go_from_official() {
  local os arch go_os go_arch archive url tmp_dir
  os="$(detect_os)"
  arch="$(detect_arch)"

  case "$os" in
    linux) go_os="linux" ;;
    darwin) go_os="darwin" ;;
    *) echo "Unsupported OS for Go official download: $os"; return 1 ;;
  esac

  case "$arch" in
    amd64|arm64|386|armv6l) go_arch="$arch" ;;
    *) echo "Unsupported architecture for Go official download: $arch"; return 1 ;;
  esac

  archive="go${GO_VERSION}.${go_os}-${go_arch}.tar.gz"
  url="https://go.dev/dl/${archive}"
  tmp_dir="$(mktemp -d)"
  mkdir -p "$GO_INSTALL_ROOT"

  echo "Downloading Go from $url"
  curl -fL "$url" -o "$tmp_dir/$archive"
  rm -rf "$GO_INSTALL_ROOT/go"
  tar -C "$GO_INSTALL_ROOT" -xzf "$tmp_dir/$archive"
  rm -rf "$tmp_dir"
  export PATH="$GO_INSTALL_ROOT/go/bin:$PATH"
  command -v go >/dev/null 2>&1
}

ensure_go() {
  command -v go >/dev/null 2>&1 && return
  local manager
  manager="$(detect_pkg_manager)"
  case "$manager" in
    apt) install_packages "$manager" golang-go || true ;;
    dnf|yum|pacman|zypper|apk|brew) install_packages "$manager" go || true ;;
    *) ;;
  esac
  command -v go >/dev/null 2>&1 && return
  install_go_from_official
  command -v go >/dev/null 2>&1
}

ensure_vips_optional() {
  command -v vips >/dev/null 2>&1 && return
  local manager
  manager="$(detect_pkg_manager)"
  case "$manager" in
    apt) install_packages "$manager" libvips-tools || true ;;
    dnf|yum|pacman|zypper|apk|brew) install_packages "$manager" vips || true ;;
    *) ;;
  esac
  if ! command -v vips >/dev/null 2>&1; then
    echo "warning: vips is missing. sample test will run with apply_crop=false."
  fi
}

create_or_repair_venv() {
  if [[ -d "$VENV_DIR" && ! -x "$VENV_DIR/bin/python3" ]]; then
    rm -rf "$VENV_DIR"
  fi
  if [[ -d "$VENV_DIR" ]]; then
    return
  fi

  echo "Creating virtualenv at $VENV_DIR"
  # Prefer virtualenv to avoid ensurepip failures on minimal/system Python builds.
  if python3 -m pip install --upgrade pip virtualenv && python3 -m virtualenv "$VENV_DIR"; then
    return
  fi

  echo "virtualenv creation failed. Trying package fix + fallback."
  local manager
  manager="$(detect_pkg_manager)"
  case "$manager" in
    apt)
      install_packages "$manager" python3-venv || install_packages "$manager" python3-full || true
      ;;
    zypper)
      install_packages "$manager" python3-virtualenv || true
      ;;
    apk)
      install_packages "$manager" py3-virtualenv || true
      ;;
    *)
      ;;
  esac

  if python3 -m pip install --upgrade pip virtualenv && python3 -m virtualenv "$VENV_DIR"; then
    return
  fi

  # Last resort: venv module (can emit ensurepip errors on some distros).
  python3 -m venv "$VENV_DIR"
}

health_ok() {
  local port="$1"
  curl -fsS "http://$HOST:$port/healthz" >/dev/null 2>&1
}

port_is_listening() {
  local port="$1"

  if command -v ss >/dev/null 2>&1; then
    ss -ltn "( sport = :$port )" 2>/dev/null | grep -q ":$port" && return 0
  fi

  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 && return 0
  fi

  if command -v netstat >/dev/null 2>&1; then
    netstat -an 2>/dev/null | grep -E "[\.\:]$port[[:space:]].*LISTEN" >/dev/null 2>&1 && return 0
  fi

  return 1
}

find_available_port() {
  local start="$1"
  local max_tries="${2:-50}"
  local candidate="$start"
  local i
  for ((i=0; i<max_tries; i++)); do
    if ! port_is_listening "$candidate"; then
      echo "$candidate"
      return 0
    fi
    candidate=$((candidate + 1))
  done
  return 1
}

run_sample_test() {
  if [[ "$RUN_SAMPLE_TEST" != "1" ]]; then
    echo "Sample test disabled (RUN_SAMPLE_TEST=$RUN_SAMPLE_TEST)"
    return 0
  fi

  mapfile -t IMAGES < <(find "$SAMPLE_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | sort)
  if [[ ${#IMAGES[@]} -eq 0 ]]; then
    echo "No sample images found in $SAMPLE_DIR"
    return 1
  fi

  local apply_crop=0
  if command -v vips >/dev/null 2>&1 && [[ "$SAMPLE_APPLY_CROP" == "1" ]]; then
    apply_crop=1
  fi

  python3 - "$PAYLOAD_FILE" "$OUTPUT_IMAGE" "$apply_crop" "${IMAGES[@]}" <<'PY'
import json
import sys

payload_path = sys.argv[1]
output_image = sys.argv[2]
apply_crop = sys.argv[3] == "1"
image_paths = sys.argv[4:]

payload = {
    "image_paths": image_paths,
    "preferred_ratio": "16:9",
    "max_analysis_size": 512,
    "apply_crop": apply_crop,
    "quality": 85,
    "return_candidates": True,
}
if apply_crop:
    payload["output_path"] = output_image

with open(payload_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
PY

  curl -fsS -X POST "http://$HOST:$PORT/thumbnail" \
    -H "Content-Type: application/json" \
    --data @"$PAYLOAD_FILE" >"$RESP_FILE"

  python3 - "$RESP_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)

print("Sample test result")
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

  if [[ "$apply_crop" == "1" ]]; then
    if [[ -f "$OUTPUT_IMAGE" ]]; then
      echo "Sample thumbnail generated: $OUTPUT_IMAGE"
    else
      echo "Sample test failed: thumbnail file not generated."
      return 1
    fi
  else
    echo "Sample test completed with apply_crop=false (vips missing or disabled)."
  fi

  echo "Sample response: $RESP_FILE"
}

OS_NAME="$(detect_os)"
ARCH_NAME="$(detect_arch)"
echo "Detected OS: $OS_NAME | ARCH: $ARCH_NAME"

if [[ "$OS_NAME" == "unsupported" || "$ARCH_NAME" == "unsupported" ]]; then
  echo "Unsupported OS/architecture combination."
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Stopping old server PID: $OLD_PID"
    kill "$OLD_PID" || true
    sleep 1
  fi
  rm -f "$PID_FILE"
fi

echo "Checking dependencies and auto-installing when missing..."
ensure_curl
ensure_python3
ensure_go
ensure_vips_optional
echo "Using python: $(command -v python3)"
echo "Using go: $(command -v go)"
if command -v vips >/dev/null 2>&1; then
  echo "Using vips: $(command -v vips)"
fi
if [[ -n "${ANIME_FACE_ONNX_PATH:-}" ]]; then
  if [[ -f "$ANIME_FACE_ONNX_PATH" ]]; then
    echo "Using AI face model: $ANIME_FACE_ONNX_PATH"
  else
    echo "warning: ANIME_FACE_ONNX_PATH is set but file not found: $ANIME_FACE_ONNX_PATH"
  fi
else
  echo "warning: ANIME_FACE_ONNX_PATH not set. Worker will use cascade fallback for face detection."
fi

create_or_repair_venv

source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

go build -o "$BIN_PATH" .

if health_ok "$PORT"; then
  echo "Existing server is already healthy on http://$HOST:$PORT, reusing it."
  run_sample_test
  echo "Log: $LOG_FILE"
  exit 0
fi

if port_is_listening "$PORT"; then
  echo "Port $PORT is already in use but health check failed."
  if [[ "$AUTO_PORT_FALLBACK" == "1" ]]; then
    NEW_PORT="$(find_available_port $((PORT + 1)) 50 || true)"
    if [[ -z "${NEW_PORT:-}" ]]; then
      echo "No available fallback port found."
      echo "Log: $LOG_FILE"
      exit 1
    fi
    echo "Switching to available port: $NEW_PORT"
    PORT="$NEW_PORT"
  else
    echo "Auto port fallback disabled (AUTO_PORT_FALLBACK=$AUTO_PORT_FALLBACK)."
    echo "Log: $LOG_FILE"
    exit 1
  fi
fi

echo "Starting server in background on http://$HOST:$PORT"
nohup env PORT="$PORT" THUMBNAIL_WORKER_PATH="$ROOT_DIR/thumbnail_worker.py" "$BIN_PATH" >>"$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" >"$PID_FILE"

for _ in {1..30}; do
  if curl -fsS "http://$HOST:$PORT/healthz" >/dev/null 2>&1; then
    echo "Server is running"
    echo "PID: $NEW_PID"
    echo "Health: http://$HOST:$PORT/healthz"
    run_sample_test
    echo "Log: $LOG_FILE"
    exit 0
  fi
  if ! kill -0 "$NEW_PID" 2>/dev/null; then
    echo "Server process exited before becoming healthy (PID: $NEW_PID)."
    echo "Log: $LOG_FILE"
    exit 1
  fi
  sleep 1
done

echo "Server failed to become healthy within timeout."
echo "Log: $LOG_FILE"
exit 1
