#!/usr/bin/env python3
"""Python worker for Go thumbnail service.

Contract:
- Input: ``--input-json '<json>'`` or ``--input-json-file <path>``.
- Output: exactly one JSON object on stdout with keys:
  ``crop_x``, ``crop_y``, ``crop_width``, ``crop_height``, ``method``, ``confidence``.

The worker is defensive: it always emits JSON output even when analysis fails.
Pipeline priority: YOLO -> saliency -> face cascade -> fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None

try:
    from PIL import Image
except Exception:  # noqa: BLE001
    Image = None

try:
    from ultralytics import YOLO
except Exception:  # noqa: BLE001
    YOLO = None

DEFAULT_RATIO = 16.0 / 9.0
DEFAULT_MAX_ANALYSIS_SIZE = 512
DEFAULT_YOLO_MODEL = "yolov8n.pt"

_YOLO_MODEL_CACHE: Any = None
_YOLO_MODEL_KEY = ""
_YOLO_MODEL_INIT_FAILED = False


@dataclass
class CropResult:
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    method: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_x": int(self.crop_x),
            "crop_y": int(self.crop_y),
            "crop_width": int(self.crop_width),
            "crop_height": int(self.crop_height),
            "method": self.method,
            "confidence": round(float(clamp(self.confidence, 0.0, 1.0)), 6),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thumbnail crop worker")
    parser.add_argument("--input-json", default="", help="JSON payload as string")
    parser.add_argument("--input-json-file", default="", help="Path to JSON payload")
    return parser.parse_args()


def parse_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_json:
        try:
            return json.loads(args.input_json)
        except Exception as err:  # noqa: BLE001
            log_err(f"invalid --input-json payload: {err}")
            return {}

    if args.input_json_file:
        try:
            with open(args.input_json_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as err:  # noqa: BLE001
            log_err(f"invalid --input-json-file payload: {err}")
            return {}

    log_err("missing input payload, using empty request")
    return {}


def log_err(message: str) -> None:
    print(message, file=sys.stderr)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_ratio(value: Any) -> float:
    if value is None:
        return DEFAULT_RATIO

    if isinstance(value, (int, float)) and float(value) > 0:
        return float(value)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return DEFAULT_RATIO
        if ":" in text:
            left, right = text.split(":", 1)
            try:
                lv = float(left.strip())
                rv = float(right.strip())
                if lv > 0 and rv > 0:
                    return lv / rv
            except Exception:  # noqa: BLE001
                return DEFAULT_RATIO
        try:
            val = float(text)
            if val > 0:
                return val
        except Exception:  # noqa: BLE001
            return DEFAULT_RATIO

    return DEFAULT_RATIO


def to_int(value: Any, default: int = 0) -> int:
    try:
        iv = int(value)
        return iv if iv > 0 else default
    except Exception:  # noqa: BLE001
        return default


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def parse_csv_env(name: str, default_value: str) -> list[str]:
    raw = os.getenv(name, default_value)
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def get_yolo_model() -> Any:
    global _YOLO_MODEL_CACHE
    global _YOLO_MODEL_KEY
    global _YOLO_MODEL_INIT_FAILED

    if YOLO is None:
        return None

    model_path = os.getenv("THUMBNAIL_YOLO_MODEL", DEFAULT_YOLO_MODEL).strip() or DEFAULT_YOLO_MODEL
    if _YOLO_MODEL_CACHE is not None and _YOLO_MODEL_KEY == model_path:
        return _YOLO_MODEL_CACHE
    if _YOLO_MODEL_INIT_FAILED and _YOLO_MODEL_KEY == model_path:
        return None

    try:
        model = YOLO(model_path)
        _YOLO_MODEL_CACHE = model
        _YOLO_MODEL_KEY = model_path
        _YOLO_MODEL_INIT_FAILED = False
        return model
    except Exception as err:  # noqa: BLE001
        _YOLO_MODEL_CACHE = None
        _YOLO_MODEL_KEY = model_path
        _YOLO_MODEL_INIT_FAILED = True
        log_err(f"yolo model load failed ({model_path}): {err}")
        return None


def match_class_weight(class_name: str, include_words: list[str], avoid_words: list[str]) -> float:
    name = class_name.strip().lower()
    if not name:
        return 0.55
    if any(word in name for word in avoid_words):
        return 0.15
    if any(word in name for word in include_words):
        return 1.0
    return 0.55


def detect_yolo_crop(image_bgr: np.ndarray, ratio: float, max_side: int) -> CropResult | None:
    model = get_yolo_model()
    if model is None:
        return None

    detect_conf = clamp(to_float(os.getenv("THUMBNAIL_YOLO_CONF", "0.22"), 0.22), 0.05, 0.90)
    detect_iou = clamp(to_float(os.getenv("THUMBNAIL_YOLO_IOU", "0.50"), 0.50), 0.10, 0.90)
    detect_max_det = max(1, to_int(os.getenv("THUMBNAIL_YOLO_MAX_DET", "40"), 40))
    imgsz = max(512, min(1280, max_side * 2))

    include_words = parse_csv_env(
        "THUMBNAIL_YOLO_TARGETS",
        "person,face,head,human,man,woman,boy,girl,character",
    )
    avoid_words = parse_csv_env(
        "THUMBNAIL_YOLO_AVOID",
        "text,caption,subtitle,logo,watermark",
    )

    try:
        results = model.predict(
            source=image_bgr,
            imgsz=imgsz,
            conf=detect_conf,
            iou=detect_iou,
            max_det=detect_max_det,
            verbose=False,
        )
    except Exception as err:  # noqa: BLE001
        log_err(f"yolo predict failed: {err}")
        return None

    if not results:
        return None

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    try:
        xyxy = boxes.xyxy.cpu().numpy()
    except Exception:  # noqa: BLE001
        return None

    if xyxy.size == 0:
        return None

    confs = None
    classes = None
    try:
        if getattr(boxes, "conf", None) is not None:
            confs = boxes.conf.cpu().numpy()
    except Exception:  # noqa: BLE001
        confs = None
    try:
        if getattr(boxes, "cls", None) is not None:
            classes = boxes.cls.cpu().numpy()
    except Exception:  # noqa: BLE001
        classes = None

    names_map = getattr(result, "names", None)
    if names_map is None:
        names_map = getattr(model, "names", {})

    ih, iw = image_bgr.shape[:2]
    full_area = float(max(1, iw * ih))
    best_idx = -1
    best_score = -1.0

    for idx in range(xyxy.shape[0]):
        x1, y1, x2, y2 = [float(v) for v in xyxy[idx]]
        x1 = clamp(x1, 0.0, float(max(0, iw - 1)))
        y1 = clamp(y1, 0.0, float(max(0, ih - 1)))
        x2 = clamp(x2, 0.0, float(max(0, iw - 1)))
        y2 = clamp(y2, 0.0, float(max(0, ih - 1)))
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        area_ratio = (bw * bh) / full_area
        if area_ratio < 0.003:
            continue

        det_conf = float(confs[idx]) if confs is not None and idx < len(confs) else 0.50
        cls_name = ""
        if classes is not None and idx < len(classes):
            cls_id = int(classes[idx])
            if isinstance(names_map, dict):
                cls_name = str(names_map.get(cls_id, cls_id))
            elif isinstance(names_map, list) and 0 <= cls_id < len(names_map):
                cls_name = str(names_map[cls_id])
            else:
                cls_name = str(cls_id)

        class_weight = match_class_weight(cls_name, include_words, avoid_words)

        cx = (x1 + x2) / 2.0 / max(1.0, float(iw))
        cy = (y1 + y2) / 2.0 / max(1.0, float(ih))
        center_penalty = min(1.0, abs(cx - 0.5) * 1.3 + abs(cy - 0.45) * 0.8)
        center_score = 1.0 - center_penalty

        size_score = 1.0 - min(1.0, abs(area_ratio - 0.18) / 0.18)

        score = (
            0.55 * clamp(det_conf, 0.0, 1.0)
            + 0.20 * class_weight
            + 0.15 * center_score
            + 0.10 * size_score
        )
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0:
        return None

    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy[best_idx]]
    x1 = max(0, min(x1, iw - 1))
    y1 = max(0, min(y1, ih - 1))
    x2 = max(0, min(x2, iw - 1))
    y2 = max(0, min(y2, ih - 1))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    pad_x = int(round(bw * 0.28))
    pad_y = int(round(bh * 0.48))
    x = max(0, x1 - pad_x)
    y = max(0, y1 - int(round(pad_y * 0.70)))
    w = min(iw - x, bw + (2 * pad_x))
    h = min(ih - y, bh + (2 * pad_y))

    crop = expand_rect_to_ratio(x, y, w, h, ratio, iw, ih)
    crop.method = "yolo"
    crop.confidence = clamp(best_score, 0.05, 0.98)
    return crop


def safe_imread(path: str) -> np.ndarray | None:
    if cv2 is None:
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def resolve_dimensions(payload: dict[str, Any]) -> tuple[int, int]:
    width = to_int(payload.get("image_width"), 0)
    height = to_int(payload.get("image_height"), 0)

    if width > 0 and height > 0:
        return width, height

    image_path = str(payload.get("image_path") or "").strip()
    if not image_path:
        return width, height

    if Image is not None:
        try:
            with Image.open(image_path) as img:
                iw, ih = img.size
            if width <= 0:
                width = iw
            if height <= 0:
                height = ih
        except Exception as err:  # noqa: BLE001
            log_err(f"cannot read image dimension: {err}")
    return max(width, 0), max(height, 0)


def fit_ratio_size(desired_w: float, desired_h: float, image_w: int, image_h: int, ratio: float) -> tuple[int, int]:
    width = max(1.0, desired_w)
    height = max(1.0, desired_h)

    if width / height > ratio:
        width = height * ratio
    else:
        height = width / ratio

    if width > image_w:
        width = float(image_w)
        height = width / ratio
    if height > image_h:
        height = float(image_h)
        width = height * ratio

    wi = max(1, min(int(round(width)), image_w))
    hi = max(1, min(int(round(height)), image_h))

    if wi / hi > ratio:
        wi = max(1, min(int(round(hi * ratio)), image_w))
    else:
        hi = max(1, min(int(round(wi / ratio)), image_h))

    return wi, hi


def fallback_crop(image_w: int, image_h: int, ratio: float) -> CropResult:
    if image_w <= 0 or image_h <= 0:
        return CropResult(0, 0, max(0, image_w), max(0, image_h), "fallback", 0.0)

    target_w = image_w
    target_h = int(round(target_w / ratio))
    top_bias = 0.18

    if target_h <= image_h:
        y = int(round(image_h * top_bias))
        if y + target_h > image_h:
            y = max(0, image_h - target_h)
        return CropResult(0, y, target_w, target_h, "fallback", 0.0)

    target_h = image_h
    target_w = int(round(target_h * ratio))
    if target_w > image_w:
        target_w = image_w
        target_h = int(round(target_w / ratio))
        target_h = min(target_h, image_h)

    x = max(0, (image_w - target_w) // 2)
    return CropResult(x, 0, target_w, target_h, "fallback", 0.0)


def ensure_in_bounds(x: int, y: int, w: int, h: int, image_w: int, image_h: int, ratio: float) -> CropResult:
    if image_w <= 0 or image_h <= 0:
        return CropResult(0, 0, max(0, w), max(0, h), "fallback", 0.0)

    if w <= 0 or h <= 0:
        return fallback_crop(image_w, image_h, ratio)

    w = min(max(w, 1), image_w)
    h = min(max(h, 1), image_h)
    w, h = fit_ratio_size(float(w), float(h), image_w, image_h, ratio)

    x = min(max(x, 0), image_w - w)
    y = min(max(y, 0), image_h - h)

    return CropResult(x, y, w, h, "", 0.0)


def expand_rect_to_ratio(x: int, y: int, w: int, h: int, ratio: float, image_w: int, image_h: int) -> CropResult:
    cx = x + w / 2.0
    cy = y + h / 2.0
    if w / max(h, 1) > ratio:
        h = int(round(w / ratio))
    else:
        w = int(round(h * ratio))

    out_x = int(round(cx - (w / 2.0)))
    out_y = int(round(cy - (h / 2.0)))
    return ensure_in_bounds(out_x, out_y, w, h, image_w, image_h, ratio)


def resize_for_analysis(image_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    if cv2 is None:
        return image_bgr, 1.0
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return image_bgr, 1.0
    if max(h, w) <= max_side:
        return image_bgr, 1.0
    scale = float(max_side) / float(max(h, w))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def detect_saliency_crop(image_bgr: np.ndarray, ratio: float, max_side: int) -> CropResult | None:
    if cv2 is None:
        return None
    if not hasattr(cv2, "saliency"):
        return None

    analysis_img, scale = resize_for_analysis(image_bgr, max_side)
    detector = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, saliency_map = detector.computeSaliency(analysis_img)
    if not ok or saliency_map is None:
        return None

    saliency = np.asarray(saliency_map, dtype=np.float32)
    if saliency.size == 0:
        return None

    saliency = cv2.GaussianBlur(saliency, (7, 7), 0)
    threshold = float(np.quantile(saliency, 0.90))
    mask = (saliency >= threshold).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    ah, aw = analysis_img.shape[:2]
    full_area = float(max(1, aw * ah))
    top = None
    top_score = -1.0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < full_area * 0.004:
            continue
        center_y = y + (h / 2.0)
        top_bias = 1.0 - 0.20 * (center_y / max(1.0, float(ah)))
        score = area * top_bias
        if score > top_score:
            top_score = score
            top = (x, y, w, h)

    if top is None:
        return None

    x, y, w, h = top
    pad_x = int(round(w * 0.25))
    pad_y = int(round(h * 0.25))
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(aw - x, w + (2 * pad_x))
    h = min(ah - y, h + (2 * pad_y))

    if scale <= 0:
        scale = 1.0
    inv = 1.0 / scale
    ox = int(round(x * inv))
    oy = int(round(y * inv))
    ow = int(round(w * inv))
    oh = int(round(h * inv))

    ih, iw = image_bgr.shape[:2]
    crop = expand_rect_to_ratio(ox, oy, ow, oh, ratio, iw, ih)
    crop.method = "saliency"

    rect = saliency[y : y + h, x : x + w]
    saliency_mean = float(rect.mean()) if rect.size > 0 else 0.0
    area_ratio = float((w * h) / max(1.0, full_area))
    confidence = 0.50 + (0.45 * saliency_mean) - (0.12 * abs(area_ratio - 0.25))
    crop.confidence = clamp(confidence, 0.05, 0.95)
    return crop


def resolve_cascade_path() -> str:
    if cv2 is None:
        return ""
    custom = os.getenv("ANIME_FACE_CASCADE_PATH", "").strip()
    if custom and os.path.isfile(custom):
        return custom
    default = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if os.path.isfile(default):
        return default
    return ""


def detect_face_crop(image_bgr: np.ndarray, ratio: float, max_side: int) -> CropResult | None:
    if cv2 is None:
        return None
    cascade_path = resolve_cascade_path()
    if not cascade_path:
        return None

    classifier = cv2.CascadeClassifier(cascade_path)
    if classifier.empty():
        return None

    analysis_img, scale = resize_for_analysis(image_bgr, max_side)
    gray = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(24, 24),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if faces is None or len(faces) == 0:
        return None

    ah, aw = analysis_img.shape[:2]
    full_area = float(max(1, aw * ah))
    best = None
    best_score = -1.0
    for x, y, w, h in faces:
        area = float(w * h)
        center_y = y + (h / 2.0)
        top_bias = 1.0 - 0.25 * (center_y / max(1.0, float(ah)))
        score = area * top_bias
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        return None

    x, y, w, h = best
    cx = x + (w / 2.0)
    cy = y + (h * 0.42)
    panel_h = max(float(h) * 3.6, float(ah) * 0.34)
    panel_w = panel_h * ratio

    out_x = int(round(cx - (panel_w / 2.0)))
    out_y = int(round(cy - (panel_h * 0.45)))
    out_w = int(round(panel_w))
    out_h = int(round(panel_h))

    if scale <= 0:
        scale = 1.0
    inv = 1.0 / scale
    ox = int(round(out_x * inv))
    oy = int(round(out_y * inv))
    ow = int(round(out_w * inv))
    oh = int(round(out_h * inv))

    ih, iw = image_bgr.shape[:2]
    crop = ensure_in_bounds(ox, oy, ow, oh, iw, ih, ratio)
    crop.method = "face"
    area_ratio = float((w * h) / max(1.0, full_area))
    crop.confidence = clamp(0.70 + min(0.25, area_ratio * 10.0), 0.10, 0.96)
    return crop


def choose_best_crop(image_bgr: np.ndarray, ratio: float, max_analysis_size: int) -> CropResult:
    ih, iw = image_bgr.shape[:2]
    fallback = fallback_crop(iw, ih, ratio)

    yolo = None
    saliency = None
    face = None
    try:
        yolo = detect_yolo_crop(image_bgr, ratio, max_analysis_size)
    except Exception as err:  # noqa: BLE001
        log_err(f"yolo detection failed: {err}")

    try:
        saliency = detect_saliency_crop(image_bgr, ratio, max_analysis_size)
    except Exception as err:  # noqa: BLE001
        log_err(f"saliency detection failed: {err}")

    try:
        face = detect_face_crop(image_bgr, ratio, max_analysis_size)
    except Exception as err:  # noqa: BLE001
        log_err(f"face detection failed: {err}")

    if yolo:
        # YOLO is the primary detector; only override when an alternative is much stronger.
        if saliency and saliency.confidence >= yolo.confidence + 0.18 and saliency.confidence >= 0.40:
            return saliency
        if face and face.confidence >= yolo.confidence + 0.20 and face.confidence >= 0.50:
            return face
        return yolo

    if saliency and saliency.confidence >= 0.28:
        if face and face.confidence > saliency.confidence + 0.12:
            return face
        return saliency

    if face:
        return face

    if saliency:
        return saliency

    return fallback


def generate_crop(payload: dict[str, Any]) -> CropResult:
    ratio = parse_ratio(payload.get("preferred_ratio"))
    max_analysis_size = to_int(payload.get("max_analysis_size"), DEFAULT_MAX_ANALYSIS_SIZE)
    if max_analysis_size <= 0:
        max_analysis_size = DEFAULT_MAX_ANALYSIS_SIZE

    image_path = str(payload.get("image_path") or "").strip()
    width, height = resolve_dimensions(payload)
    fallback = fallback_crop(width, height, ratio)

    if not image_path:
        return fallback
    if not os.path.isfile(image_path):
        log_err(f"image file not found: {image_path}")
        return fallback

    image_bgr = safe_imread(image_path)
    if image_bgr is None:
        log_err(f"cannot decode image: {image_path}")
        return fallback

    ih, iw = image_bgr.shape[:2]
    if width <= 0 or height <= 0:
        width, height = iw, ih

    best = choose_best_crop(image_bgr, ratio, max_analysis_size)
    bounded = ensure_in_bounds(best.crop_x, best.crop_y, best.crop_width, best.crop_height, width, height, ratio)
    bounded.method = best.method or "fallback"
    bounded.confidence = clamp(best.confidence, 0.0, 1.0)
    return bounded


def main() -> int:
    args = parse_args()
    payload = parse_payload(args)
    result = generate_crop(payload)
    print(json.dumps(result.to_dict(), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
