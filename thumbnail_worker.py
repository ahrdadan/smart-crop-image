#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None


DEFAULT_RATIO = 16.0 / 9.0
DEFAULT_MAX_ANALYSIS_SIZE = 512
SALIENCY_ACCEPT_THRESHOLD = 0.6
FACE_ACCEPT_THRESHOLD = 0.5


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _parse_ratio(value: Any) -> float:
    if value is None:
        return DEFAULT_RATIO
    if isinstance(value, (int, float)):
        ratio = float(value)
        return ratio if ratio > 0 else DEFAULT_RATIO
    text = str(value).strip()
    if ":" in text:
        left, right = text.split(":", 1)
        try:
            l = float(left)
            r = float(right)
            if l > 0 and r > 0:
                return l / r
        except Exception:
            return DEFAULT_RATIO
    try:
        ratio = float(text)
        return ratio if ratio > 0 else DEFAULT_RATIO
    except Exception:
        return DEFAULT_RATIO


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _read_input() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input-json", type=str, default=None)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--preferred-ratio", type=str, default=None)
    parser.add_argument("--max-analysis-size", type=int, default=None)
    args, _unknown = parser.parse_known_args()

    payload: Dict[str, Any] = {}
    if args.input_json:
        text = args.input_json.strip()
        if text.startswith("{"):
            payload = json.loads(text)
        elif os.path.exists(text):
            with open(text, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        else:
            payload = {}
    else:
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            payload = json.loads(stdin_text)

    if args.image_path is not None:
        payload["image_path"] = args.image_path
    if args.image_width is not None:
        payload["image_width"] = args.image_width
    if args.image_height is not None:
        payload["image_height"] = args.image_height
    if args.preferred_ratio is not None:
        payload["preferred_ratio"] = args.preferred_ratio
    if args.max_analysis_size is not None:
        payload["max_analysis_size"] = args.max_analysis_size

    return payload


def _load_image_and_dims(
    image_path: Optional[str], declared_w: int, declared_h: int
) -> Tuple[Optional[np.ndarray], int, int]:
    img: Optional[np.ndarray] = None
    if image_path and cv2 is not None:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is not None:
        h, w = img.shape[:2]
        return img, int(w), int(h)

    if image_path and Image is not None:
        try:
            with Image.open(image_path) as pil_img:
                w, h = pil_img.size
            return None, int(w), int(h)
        except Exception:
            pass

    return None, max(0, declared_w), max(0, declared_h)


def _resize_for_analysis(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image
    scale = float(max_side) / float(max(h, w))
    if scale >= 1.0:
        return image
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _normalize_map(src: np.ndarray) -> np.ndarray:
    arr = src.astype(np.float32)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx <= mn + 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def _fallback_saliency_map(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return _normalize_map(mag)


def _fit_ratio_size(
    desired_w: float, desired_h: float, image_w: int, image_h: int, ratio: float
) -> Tuple[int, int]:
    w = max(1.0, desired_w)
    h = max(1.0, desired_h)
    if w / h > ratio:
        w = h * ratio
    else:
        h = w / ratio

    if w > image_w:
        w = float(image_w)
        h = w / ratio
    if h > image_h:
        h = float(image_h)
        w = h * ratio

    wi = max(1, min(image_w, int(round(w))))
    hi = max(1, min(image_h, int(round(h))))

    if wi / hi > ratio:
        wi = max(1, min(image_w, int(round(hi * ratio))))
    else:
        hi = max(1, min(image_h, int(round(wi / ratio))))

    return wi, hi


def _crop_from_center(
    cx: float, cy: float, crop_w: int, crop_h: int, image_w: int, image_h: int
) -> Tuple[int, int, int, int]:
    x = int(round(cx - crop_w / 2.0))
    y = int(round(cy - crop_h / 2.0))

    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + crop_w > image_w:
        x = max(0, image_w - crop_w)
    if y + crop_h > image_h:
        y = max(0, image_h - crop_h)

    return x, y, crop_w, crop_h


def _expand_bbox_to_ratio(
    x0: int, y0: int, x1: int, y1: int, image_w: int, image_h: int, ratio: float
) -> Tuple[int, int, int, int]:
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    cx = x0 + bw / 2.0
    cy = y0 + bh / 2.0
    target_w, target_h = _fit_ratio_size(float(bw), float(bh), image_w, image_h, ratio)
    return _crop_from_center(cx, cy, target_w, target_h, image_w, image_h)


def _score_saliency(
    saliency: np.ndarray, mask: np.ndarray, bbox_w: int, bbox_h: int
) -> float:
    h, w = saliency.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0
    if not np.any(mask):
        return 0.0

    top_mean = float(np.mean(saliency[mask]))
    global_mean = float(np.mean(saliency))
    separation = _clamp((top_mean - global_mean) / (1.0 - global_mean + 1e-6), 0.0, 1.0)

    bbox_area_ratio = float(bbox_w * bbox_h) / float(max(1, w * h))
    compactness = _clamp(1.0 - bbox_area_ratio, 0.0, 1.0)

    spread = _clamp(float(np.std(saliency)) * 2.0, 0.0, 1.0)

    score = 0.5 * separation + 0.3 * compactness + 0.2 * spread
    return _clamp(score, 0.0, 1.0)


def _saliency_crop(analysis_bgr: np.ndarray, ratio: float) -> Tuple[Tuple[int, int, int, int], float]:
    h, w = analysis_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return (0, 0, max(1, w), max(1, h)), 0.0

    if cv2 is None:
        return (0, 0, w, max(1, int(round(w / ratio)))), 0.0

    try:
        saliency_map: Optional[np.ndarray] = None
        if hasattr(cv2, "saliency") and hasattr(cv2.saliency, "StaticSaliencySpectralResidual_create"):
            detector = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, sal = detector.computeSaliency(analysis_bgr)
            if success:
                saliency_map = sal.astype(np.float32)

        if saliency_map is None:
            gray = cv2.cvtColor(analysis_bgr, cv2.COLOR_BGR2GRAY)
            saliency_map = _fallback_saliency_map(gray)

        saliency_map = _normalize_map(saliency_map)
        threshold = float(np.quantile(saliency_map, 0.70))
        mask = saliency_map >= threshold
        if not np.any(mask):
            mask = saliency_map > 0

        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            crop_h = max(1, min(h, int(round(w / ratio))))
            return (0, 0, w, crop_h), 0.0

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cx, cy, cw, ch = _expand_bbox_to_ratio(x0, y0, x1, y1, w, h, ratio)
        conf = _score_saliency(saliency_map, mask, cw, ch)
        return (cx, cy, cw, ch), conf
    except Exception:
        crop_h = max(1, min(h, int(round(w / ratio))))
        return (0, 0, w, crop_h), 0.0


def _get_face_cascade() -> Optional[Any]:
    if cv2 is None:
        return None

    candidates = []
    env_path = os.environ.get("ANIME_FACE_CASCADE_PATH", "").strip()
    if env_path:
        candidates.append(env_path)
    candidates.extend(
        [
            os.path.join("models", "lbpcascade_animeface.xml"),
            os.path.join("models", "haarcascade_frontalface_default.xml"),
            os.path.join(getattr(cv2.data, "haarcascades", ""), "haarcascade_frontalface_default.xml"),
        ]
    )

    for path in candidates:
        if not path:
            continue
        if os.path.exists(path):
            try:
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    return cascade
            except Exception:
                continue
    return None


def _face_crop(analysis_bgr: np.ndarray, ratio: float) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    if cv2 is None:
        return None, 0.0

    h, w = analysis_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return None, 0.0

    cascade = _get_face_cascade()
    if cascade is None:
        return None, 0.0

    try:
        gray = cv2.cvtColor(analysis_bgr, cv2.COLOR_BGR2GRAY)
        min_side = max(16, int(round(min(w, h) * 0.06)))
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_side, min_side),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if faces is None or len(faces) == 0:
            return None, 0.0

        faces = sorted(faces, key=lambda f: int(f[2]) * int(f[3]), reverse=True)
        fx, fy, fw, fh = [int(v) for v in faces[0]]
        cx = fx + fw / 2.0
        cy = fy + fh / 2.0

        padded_w = fw * 2.4
        padded_h = fh * 2.6
        target_w, target_h = _fit_ratio_size(padded_w, padded_h, w, h, ratio)
        crop = _crop_from_center(cx, cy, target_w, target_h, w, h)

        face_area_ratio = float(fw * fh) / float(max(1, w * h))
        conf = _clamp(0.35 + 2.0 * math.sqrt(max(0.0, face_area_ratio)), 0.0, 1.0)
        return crop, conf
    except Exception:
        return None, 0.0


def _map_crop_to_original(
    crop: Tuple[int, int, int, int], analysis_w: int, analysis_h: int, orig_w: int, orig_h: int, ratio: float
) -> Tuple[int, int, int, int]:
    ax, ay, aw, ah = crop
    if analysis_w <= 0 or analysis_h <= 0 or orig_w <= 0 or orig_h <= 0:
        return (0, 0, 0, 0)

    scale_x = float(orig_w) / float(analysis_w)
    scale_y = float(orig_h) / float(analysis_h)
    cx = (ax + aw / 2.0) * scale_x
    cy = (ay + ah / 2.0) * scale_y
    dw = max(1.0, aw * scale_x)
    dh = max(1.0, ah * scale_y)

    target_w, target_h = _fit_ratio_size(dw, dh, orig_w, orig_h, ratio)
    return _crop_from_center(cx, cy, target_w, target_h, orig_w, orig_h)


def _fallback_crop(orig_w: int, orig_h: int, ratio: float) -> Tuple[int, int, int, int]:
    if orig_w <= 0 or orig_h <= 0:
        return (0, 0, 0, 0)

    target_w = orig_w
    target_h = int(round(target_w / ratio))
    top_bias = 0.18

    if target_h <= orig_h:
        x = 0
        y = int(round(orig_h * top_bias))
        if y + target_h > orig_h:
            y = max(0, orig_h - target_h)
        return (x, y, target_w, target_h)

    target_h = orig_h
    target_w = int(round(target_h * ratio))
    if target_w > orig_w:
        target_w = orig_w
        target_h = int(round(target_w / ratio))
        target_h = min(target_h, orig_h)

    x = max(0, (orig_w - target_w) // 2)
    y = 0
    return (x, y, target_w, target_h)


def _ensure_in_bounds(
    x: int, y: int, w: int, h: int, image_w: int, image_h: int, ratio: float
) -> Tuple[int, int, int, int]:
    if image_w <= 0 or image_h <= 0:
        return (0, 0, 0, 0)

    w = max(1, min(w, image_w))
    h = max(1, min(h, image_h))
    w, h = _fit_ratio_size(float(w), float(h), image_w, image_h, ratio)

    x = max(0, min(x, image_w - w))
    y = max(0, min(y, image_h - h))
    return (int(x), int(y), int(w), int(h))


def _build_result(
    x: int, y: int, w: int, h: int, method: str, confidence: float
) -> Dict[str, Any]:
    return {
        "crop_x": int(x),
        "crop_y": int(y),
        "crop_width": int(w),
        "crop_height": int(h),
        "method": str(method),
        "confidence": float(round(_clamp(confidence, 0.0, 1.0), 4)),
    }


def main() -> None:
    payload = {}
    try:
        payload = _read_input()
    except Exception:
        payload = {}

    image_path = payload.get("image_path")
    declared_w = _safe_int(payload.get("image_width"), 0)
    declared_h = _safe_int(payload.get("image_height"), 0)
    ratio = _parse_ratio(payload.get("preferred_ratio", "16:9"))
    max_analysis_size = _safe_int(payload.get("max_analysis_size"), DEFAULT_MAX_ANALYSIS_SIZE)
    if max_analysis_size <= 0:
        max_analysis_size = DEFAULT_MAX_ANALYSIS_SIZE

    img, orig_w, orig_h = _load_image_and_dims(image_path, declared_w, declared_h)
    if orig_w <= 0 or orig_h <= 0:
        result = _build_result(0, 0, 0, 0, "fallback", 0.0)
        print(json.dumps(result, ensure_ascii=True))
        return

    method = "fallback"
    confidence = 0.0
    crop = _fallback_crop(orig_w, orig_h, ratio)

    if img is not None and cv2 is not None:
        analysis = _resize_for_analysis(img, max_analysis_size)
        ah, aw = analysis.shape[:2]

        sal_crop, sal_conf = _saliency_crop(analysis, ratio)
        mapped_sal_crop = _map_crop_to_original(sal_crop, aw, ah, orig_w, orig_h, ratio)
        mapped_sal_crop = _ensure_in_bounds(*mapped_sal_crop, orig_w, orig_h, ratio)

        if sal_conf >= SALIENCY_ACCEPT_THRESHOLD:
            crop = mapped_sal_crop
            method = "saliency"
            confidence = sal_conf
        else:
            face_crop, face_conf = _face_crop(analysis, ratio)
            if face_crop is not None:
                mapped_face_crop = _map_crop_to_original(face_crop, aw, ah, orig_w, orig_h, ratio)
                mapped_face_crop = _ensure_in_bounds(*mapped_face_crop, orig_w, orig_h, ratio)
                if face_conf >= FACE_ACCEPT_THRESHOLD:
                    crop = mapped_face_crop
                    method = "face"
                    confidence = face_conf

    crop = _ensure_in_bounds(*crop, orig_w, orig_h, ratio)
    result = _build_result(crop[0], crop[1], crop[2], crop[3], method, confidence)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
