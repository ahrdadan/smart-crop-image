#!/usr/bin/env python3
"""Chapter pair composer worker (YOLO text-aware ranking + 2-image merge).

Contract:
- Input: ``--input-json '<json>'`` or ``--input-json-file <path>``.
- Output: one JSON object to stdout:
  {
    "out_path": "...",
    "size": [1200, 675],
    "picked": [{...}, {...}]
  }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

# Default CPU mode; user can override with THUMBNAIL_YOLO_DEVICE.
if os.getenv("THUMBNAIL_YOLO_DEVICE", "cpu").strip().lower() in ("", "cpu"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

try:
    from ultralytics import YOLOE
except Exception:  # noqa: BLE001
    YOLOE = None

try:
    from ultralytics import YOLO
except Exception:  # noqa: BLE001
    YOLO = None

DEFAULT_YOLOE_MODEL = "yoloe-26s-seg.pt"
DEFAULT_SKIP_EDGES = 2
DEFAULT_GAP = 5
DEFAULT_WIDTH = 1200

_MODEL_CACHE: Any = None
_MODEL_KEY = ""


def log_err(message: str) -> None:
    print(message, file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart chapter thumbnail worker")
    parser.add_argument("--input-json", default="", help="JSON payload as string")
    parser.add_argument("--input-json-file", default="", help="Path to JSON payload")
    return parser.parse_args()


def parse_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_json:
        return json.loads(args.input_json)
    if args.input_json_file:
        with open(args.input_json_file, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return default


def get_yolo_device() -> str:
    device = os.getenv("THUMBNAIL_YOLO_DEVICE", "cpu").strip()
    return device or "cpu"


def clean_image_paths(paths: list[Any]) -> list[str]:
    out: list[str] = []
    for path in paths:
        text = str(path).strip()
        if not text:
            continue
        if os.path.isfile(text):
            out.append(text)
    return out


def list_images(folder: str, exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")) -> list[str]:
    files = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(exts):
            files.append(os.path.join(folder, fn))

    def key_nat(path: str) -> list[Any]:
        base = os.path.basename(path)
        parts: list[Any] = []
        chunk = ""
        for ch in base:
            if ch.isdigit():
                chunk += ch
                continue
            if chunk:
                parts.append(int(chunk))
                chunk = ""
            parts.append(ch.lower())
        if chunk:
            parts.append(int(chunk))
        return parts

    return sorted(files, key=key_nat)


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def cover_resize(img: Image.Image, target_w: int, target_h: int, centering=(0.5, 0.5)) -> Image.Image:
    img = img.convert("RGB")
    return ImageOps.fit(img, (target_w, target_h), method=Image.LANCZOS, centering=centering)


def window_score(bgr_patch: np.ndarray) -> float:
    h, w = bgr_patch.shape[:2]
    max_side = 360
    scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
    if scale != 1.0:
        bgr_patch = cv2.resize(bgr_patch, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0
    var = float(gray.var()) / (255.0**2)

    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].mean() / 255.0

    bright = gray.mean() / 255.0
    white_ratio = float((gray > 245).mean())

    blank_penalty = 0.0
    if white_ratio > 0.55 and edge_density < 0.03:
        blank_penalty += (white_ratio - 0.55) * 2.0
    if bright < 0.08:
        blank_penalty += (0.08 - bright) * 2.0

    return float((2.2 * edge_density) + (1.3 * var) + (0.4 * sat) - (1.8 * blank_penalty))


def best_thumbnail_crop(
    pil_img: Image.Image,
    target_ratio: float = 3 / 4,
    min_window_h: int = 700,
    stride_ratio: float = 0.18,
    top_bias: bool = True,
) -> tuple[Image.Image, float, tuple[int, int, int, int]]:
    bgr = pil_to_bgr(pil_img)
    height, width = bgr.shape[:2]

    win_w = width
    win_h = int(round(win_w / target_ratio))
    win_h = max(win_h, min_window_h)
    win_h = min(win_h, height)

    if height <= int(win_h * 1.05):
        y0 = max(0, (height - win_h) // 2)
        y1 = y0 + win_h
        patch = bgr[y0:y1, 0:width]
        return bgr_to_pil(patch), window_score(patch), (0, y0, width, y1)

    stride = max(32, int(win_h * stride_ratio))
    candidates: list[tuple[float, int]] = []

    for y0 in range(0, height - win_h + 1, stride):
        y1 = y0 + win_h
        patch = bgr[y0:y1, 0:width]
        score = window_score(patch)
        if top_bias:
            score -= (y0 / max(1, (height - win_h))) * 0.15
        candidates.append((score, y0))

    for y0 in [0, int(0.08 * (height - win_h)), int(0.15 * (height - win_h))]:
        y0 = int(np.clip(y0, 0, height - win_h))
        patch = bgr[y0 : y0 + win_h, 0:width]
        score = window_score(patch)
        if top_bias:
            score -= (y0 / max(1, (height - win_h))) * 0.15
        candidates.append((score, y0))

    best_score, best_y0 = max(candidates, key=lambda item: item[0])
    best_y1 = best_y0 + win_h
    best_patch = bgr[best_y0:best_y1, 0:width]
    return bgr_to_pil(best_patch), float(best_score), (0, int(best_y0), width, int(best_y1))


def load_text_detector(model_name: str) -> Any:
    global _MODEL_CACHE
    global _MODEL_KEY

    if _MODEL_CACHE is not None and _MODEL_KEY == model_name:
        return _MODEL_CACHE

    if YOLOE is not None:
        model = YOLOE(model_name)
        if hasattr(model, "set_classes"):
            model.set_classes(["text", "caption", "subtitle", "notice", "watermark", "logo"])
        _MODEL_CACHE = model
        _MODEL_KEY = model_name
        return model

    if YOLO is not None:
        # Fallback when YOLOE class is unavailable.
        model = YOLO(model_name)
        _MODEL_CACHE = model
        _MODEL_KEY = model_name
        return model

    raise RuntimeError("Ultralytics is not available in current environment.")


def yolo_text_ratio(model: Any, pil_img: Image.Image, imgsz: int = 960, conf: float = 0.20, iou: float = 0.50, max_det: int = 300) -> tuple[float, int]:
    width, height = pil_img.size
    results = model.predict(
        pil_img,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=get_yolo_device(),
        verbose=False,
    )
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return 0.0, 0

    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ratio = float(areas.sum() / max(1.0, (width * height)))
    return ratio, int(len(boxes))


def pick_top_k_pages_yolo(
    image_paths: list[str],
    k: int,
    skip_edges: int,
    text_ratio_hard: float,
    text_ratio_soft: float,
    middle_strength: float,
    target_ratio_for_crop: float,
    yolo_model_name: str,
) -> list[dict[str, Any]]:
    detector = load_text_detector(yolo_model_name)
    total_pages = len(image_paths)
    mid = (total_pages - 1) / 2.0

    scored: list[dict[str, Any]] = []
    for idx, path in enumerate(image_paths):
        try:
            pil_img = Image.open(path)
        except Exception as err:  # noqa: BLE001
            log_err(f"skip {path}: {err}")
            continue

        text_ratio, text_count = yolo_text_ratio(detector, pil_img)
        crop_pil, visual_score, bbox = best_thumbnail_crop(pil_img, target_ratio=target_ratio_for_crop)

        dist = abs(idx - mid) / max(1e-6, mid)
        middle_bonus = (1.0 - dist) * middle_strength

        text_bonus = 0.20 * max(0.0, (0.15 - text_ratio) / 0.15)
        text_penalty = 0.0
        if text_ratio > text_ratio_soft:
            text_penalty += (text_ratio - text_ratio_soft) * 1.6

        in_edge = (idx < skip_edges) or (idx >= total_pages - skip_edges)
        edge_penalty = 0.35 if in_edge else 0.0
        if text_ratio > text_ratio_hard and in_edge:
            edge_penalty += 0.60

        hard_reject = text_ratio > text_ratio_hard
        total = float(visual_score + middle_bonus + text_bonus - text_penalty - edge_penalty)

        scored.append(
            {
                "idx": idx,
                "path": path,
                "total": total,
                "vsc": float(visual_score),
                "text_ratio": float(text_ratio),
                "text_boxes": int(text_count),
                "bbox": bbox,
                "hard_reject": bool(hard_reject),
                "in_edge": bool(in_edge),
                "crop_pil": crop_pil,
            }
        )

    if not scored:
        raise RuntimeError("Tidak ada gambar yang bisa diproses.")

    scored_sorted = sorted(scored, key=lambda item: (item["hard_reject"], -item["total"]))
    return scored_sorted[:k]


def merge_two_images_to_16x9(
    left_img: Image.Image,
    right_img: Image.Image,
    out_path: str,
    width: int,
    gap: int,
    background: tuple[int, int, int] = (0, 0, 0),
    centering: tuple[float, float] = (0.5, 0.5),
) -> tuple[str, tuple[int, int]]:
    out_w = width
    out_h = int(round(out_w * 9 / 16))
    slot_w = (out_w - gap) // 2
    slot_h = out_h

    left = cover_resize(left_img, slot_w, slot_h, centering=centering)
    right = cover_resize(right_img, slot_w, slot_h, centering=centering)

    canvas = Image.new("RGB", (out_w, out_h), background)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (slot_w + gap, 0))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path, quality=95)
    return out_path, (out_w, out_h)


def make_seo_16x9_from_top2(
    image_paths: list[str],
    out_path: str,
    yolo_model_name: str,
    skip_edges: int,
    gap: int,
    width: int,
    centering: tuple[float, float] = (0.5, 0.35),
) -> dict[str, Any]:
    top2 = pick_top_k_pages_yolo(
        image_paths=image_paths,
        k=2,
        skip_edges=skip_edges,
        text_ratio_hard=0.55,
        text_ratio_soft=0.25,
        middle_strength=0.40,
        target_ratio_for_crop=3 / 4,
        yolo_model_name=yolo_model_name,
    )

    if len(top2) == 1:
        top2 = [top2[0], top2[0]]

    out_file, size = merge_two_images_to_16x9(
        left_img=top2[0]["crop_pil"],
        right_img=top2[1]["crop_pil"],
        out_path=out_path,
        width=width,
        gap=gap,
        background=(0, 0, 0),
        centering=centering,
    )

    return {
        "out_path": out_file,
        "size": [int(size[0]), int(size[1])],
        "picked": [
            {key: top2[0][key] for key in ["idx", "path", "total", "text_ratio", "in_edge", "bbox"]},
            {key: top2[1][key] for key in ["idx", "path", "total", "text_ratio", "in_edge", "bbox"]},
        ],
    }


def run(payload: dict[str, Any]) -> dict[str, Any]:
    image_paths = clean_image_paths(payload.get("image_paths") or [])
    if not image_paths:
        image_dir = str(payload.get("image_dir") or "").strip()
        if image_dir and os.path.isdir(image_dir):
            image_paths = list_images(image_dir)
    if not image_paths:
        raise RuntimeError("image_paths kosong atau tidak valid.")

    out_path = str(payload.get("output_path") or "").strip()
    if not out_path:
        raise RuntimeError("output_path wajib diisi untuk smart pair composer.")

    yolo_model_name = str(payload.get("yolo_model_name") or os.getenv("THUMBNAIL_YOLO_MODEL", DEFAULT_YOLOE_MODEL)).strip()
    if not yolo_model_name:
        yolo_model_name = DEFAULT_YOLOE_MODEL

    skip_edges = max(0, to_int(payload.get("skip_edges"), DEFAULT_SKIP_EDGES))
    gap = max(0, to_int(payload.get("gap"), DEFAULT_GAP))
    width = max(320, to_int(payload.get("width"), DEFAULT_WIDTH))

    return make_seo_16x9_from_top2(
        image_paths=image_paths,
        out_path=out_path,
        yolo_model_name=yolo_model_name,
        skip_edges=skip_edges,
        gap=gap,
        width=width,
    )


def main() -> int:
    try:
        payload = parse_payload(parse_args())
        info = run(payload)
        print(json.dumps(info, separators=(",", ":")))
        return 0
    except Exception as err:  # noqa: BLE001
        log_err(f"smart_thumb failed: {err}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
