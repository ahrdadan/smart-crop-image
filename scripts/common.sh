#!/usr/bin/env bash

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
  if command -v apt-get >/dev/null 2>&1; then
    echo "apt"
    return
  fi
  if command -v dnf >/dev/null 2>&1; then
    echo "dnf"
    return
  fi
  if command -v yum >/dev/null 2>&1; then
    echo "yum"
    return
  fi
  if command -v pacman >/dev/null 2>&1; then
    echo "pacman"
    return
  fi
  if command -v zypper >/dev/null 2>&1; then
    echo "zypper"
    return
  fi
  if command -v apk >/dev/null 2>&1; then
    echo "apk"
    return
  fi
  if command -v brew >/dev/null 2>&1; then
    echo "brew"
    return
  fi
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

  if [[ ${#packages[@]} -eq 0 ]]; then
    return
  fi

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
  if command -v curl >/dev/null 2>&1; then
    return
  fi
  local manager
  manager="$(detect_pkg_manager)"
  case "$manager" in
    apt|dnf|yum|pacman|zypper|apk|brew)
      install_packages "$manager" curl
      ;;
    *)
      echo "curl not found and cannot auto-install."
      return 1
      ;;
  esac
  command -v curl >/dev/null 2>&1
}

ensure_python3() {
  if command -v python3 >/dev/null 2>&1; then
    return
  fi

  local manager
  manager="$(detect_pkg_manager)"
  case "$manager" in
    apt)
      install_packages "$manager" python3 python3-venv python3-pip
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
  local os arch version go_os go_arch archive url tmp_dir install_root
  os="$(detect_os)"
  arch="$(detect_arch)"
  version="${GO_VERSION:-1.22.12}"
  install_root="${GO_INSTALL_ROOT:-$HOME/.local}"

  case "$os" in
    linux) go_os="linux" ;;
    darwin) go_os="darwin" ;;
    *)
      echo "Unsupported OS for Go official download: $os"
      return 1
      ;;
  esac

  case "$arch" in
    amd64|arm64|386|armv6l) go_arch="$arch" ;;
    *)
      echo "Unsupported architecture for Go official download: $arch"
      return 1
      ;;
  esac

  archive="go${version}.${go_os}-${go_arch}.tar.gz"
  url="https://go.dev/dl/${archive}"
  tmp_dir="$(mktemp -d)"
  mkdir -p "$install_root"

  echo "Downloading Go from $url"
  if ! curl -fL "$url" -o "$tmp_dir/$archive"; then
    echo "Failed downloading Go archive."
    rm -rf "$tmp_dir"
    return 1
  fi

  rm -rf "$install_root/go"
  tar -C "$install_root" -xzf "$tmp_dir/$archive"
  rm -rf "$tmp_dir"

  export PATH="$install_root/go/bin:$PATH"
  if ! command -v go >/dev/null 2>&1; then
    echo "Go installation finished but go binary still not found in PATH."
    return 1
  fi

  echo "Go installed at $install_root/go"
  echo "If needed, add this to your shell profile: export PATH=\"$install_root/go/bin:\$PATH\""
}

ensure_go() {
  if command -v go >/dev/null 2>&1; then
    return
  fi

  local manager
  manager="$(detect_pkg_manager)"
  case "$manager" in
    apt)
      install_packages "$manager" golang-go
      ;;
    dnf|yum)
      install_packages "$manager" golang
      ;;
    pacman)
      install_packages "$manager" go
      ;;
    zypper)
      install_packages "$manager" go
      ;;
    apk)
      install_packages "$manager" go
      ;;
    brew)
      install_packages "$manager" go
      ;;
    none)
      ;;
    *)
      ;;
  esac

  if command -v go >/dev/null 2>&1; then
    return
  fi

  install_go_from_official
  command -v go >/dev/null 2>&1
}

ensure_vips_optional() {
  if command -v vips >/dev/null 2>&1; then
    return
  fi

  local manager
  manager="$(detect_pkg_manager)"
  case "$manager" in
    apt)
      install_packages "$manager" libvips-tools || true
      ;;
    dnf|yum|pacman|zypper|apk|brew)
      install_packages "$manager" vips || true
      ;;
    *)
      ;;
  esac

  if ! command -v vips >/dev/null 2>&1; then
    echo "warning: vips is still missing. apply_crop=true will fail until libvips CLI is installed."
  fi
}

