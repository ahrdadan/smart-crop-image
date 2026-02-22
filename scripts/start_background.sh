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
SERVER_LOG="${SERVER_LOG:-$LOG_DIR/server.log}"
START_LOG="${START_LOG:-$LOG_DIR/start_background_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$LOG_DIR" "$RUN_DIR" "$BIN_DIR"
exec > >(tee -a "$START_LOG") 2>&1

source "$ROOT_DIR/scripts/common.sh"

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

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

go build -o "$BIN_PATH" .

echo "Starting server in background on http://$HOST:$PORT"
nohup env PORT="$PORT" THUMBNAIL_WORKER_PATH="$ROOT_DIR/thumbnail_worker.py" "$BIN_PATH" >"$SERVER_LOG" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" >"$PID_FILE"

for _ in {1..30}; do
  if curl -fsS "http://$HOST:$PORT/healthz" >/dev/null 2>&1; then
    echo "Server is running"
    echo "PID: $NEW_PID"
    echo "Health: http://$HOST:$PORT/healthz"
    echo "Log: $SERVER_LOG"
    echo "Start log: $START_LOG"
    exit 0
  fi
  sleep 1
done

echo "Server failed to become healthy within timeout. Check logs at $SERVER_LOG"
echo "Start log: $START_LOG"
exit 1
