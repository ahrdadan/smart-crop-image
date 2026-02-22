#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None

try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None

DEFAULT_RATIO = 16.0 / 9.0
DEFAULT_MAX_ANALYSIS_SIZE = 512
SALIENCY_ACCEPT_THRESHOLD = 0.6
FACE_ACCEPT_THRESHOLD = 0.5
FACE_OVERRIDE_THRESHOLD = 0.35
BALLOON_RISK_OVERRIDE_THRESHOLD = 0.45

_ONNX_FACE_SESSION: Any = None


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


def _estimate_text_density(gray: np.ndarray) -> float:
    if cv2 is None:
        return 0.0
    if gray.size == 0:
        return 0.0

    try:
        binary_inv = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            8,
        )
        num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)
        if num_labels <= 1:
            return 0.0

        h, w = gray.shape[:2]
        total_area = float(max(1, h * w))
        small_area = 0.0

        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 4 or area > 120:
                continue
            if bw <= 0 or bh <= 0:
                continue
            aspect = float(bw) / float(max(1, bh))
            if 0.2 <= aspect <= 6.0:
                # Prefer small, text-like components near edges of bubbles.
                cx = x + bw / 2.0
                cy = y + bh / 2.0
                edge_bias = abs(cx - w / 2.0) / max(1.0, w / 2.0) * 0.2 + abs(cy - h / 2.0) / max(1.0, h / 2.0) * 0.1
                small_area += area * (1.0 + edge_bias)

        density = small_area / total_area
        return _clamp(density * 6.0, 0.0, 1.0)
    except Exception:
        return 0.0


def _region_quality(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0

    white_ratio, text_density, low_contrast_penalty = _region_stats(gray)

    score = 1.0 - (0.55 * white_ratio + 0.25 * low_contrast_penalty + 0.30 * text_density)
    return _clamp(score, 0.0, 1.0)


def _region_stats(gray: np.ndarray) -> Tuple[float, float, float]:
    if gray.size == 0:
        return 0.0, 0.0, 1.0
    white_ratio = float(np.mean(gray >= 245))
    contrast = float(np.std(gray))
    low_contrast_penalty = _clamp(1.0 - contrast / 64.0, 0.0, 1.0)
    text_density = _estimate_text_density(gray)
    return white_ratio, text_density, low_contrast_penalty


def _balloon_risk(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    white_ratio, text_density, low_contrast_penalty = _region_stats(gray)
    risk = 0.60 * white_ratio + 0.60 * text_density + 0.25 * low_contrast_penalty
    return _clamp(risk, 0.0, 1.0)


def _face_region_valid(gray_region: np.ndarray) -> bool:
    if gray_region.size == 0:
        return False
    white_ratio, text_density, low_contrast_penalty = _region_stats(gray_region)
    if text_density > 0.23:
        return False
    if white_ratio > 0.86 and text_density > 0.10:
        return False
    if white_ratio > 0.92:
        return False
    if low_contrast_penalty > 0.88 and text_density > 0.10:
        return False
    return True


def _crop_region_gray(gray: np.ndarray, crop: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = crop
    x = max(0, min(x, gray.shape[1] - 1))
    y = max(0, min(y, gray.shape[0] - 1))
    w = max(1, min(w, gray.shape[1] - x))
    h = max(1, min(h, gray.shape[0] - y))
    return gray[y : y + h, x : x + w]


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
    saliency: np.ndarray, mask: np.ndarray, bbox_w: int, bbox_h: int, region_quality: float
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

    score = 0.42 * separation + 0.23 * compactness + 0.15 * spread + 0.20 * _clamp(region_quality, 0.0, 1.0)
    return _clamp(score, 0.0, 1.0)


def _best_saliency_bbox(mask: np.ndarray, saliency_map: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if cv2 is None:
        return None
    if mask.size == 0 or not np.any(mask):
        return None

    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed, connectivity=8)
    if num_labels <= 1:
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    best_idx = -1
    best_score = -1.0
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area <= 0 or w <= 0 or h <= 0:
            continue
        component = labels[y : y + h, x : x + w] == i
        mean_sal = float(np.mean(saliency_map[y : y + h, x : x + w][component]))
        score = mean_sal * math.sqrt(float(area))
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx < 0:
        return None

    x = int(stats[best_idx, cv2.CC_STAT_LEFT])
    y = int(stats[best_idx, cv2.CC_STAT_TOP])
    w = int(stats[best_idx, cv2.CC_STAT_WIDTH])
    h = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
    return (x, y, x + w - 1, y + h - 1)


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
        threshold = float(np.quantile(saliency_map, 0.75))
        mask = saliency_map >= threshold
        if not np.any(mask):
            mask = saliency_map > 0

        bbox = _best_saliency_bbox(mask, saliency_map)
        if bbox is None:
            crop_h = max(1, min(h, int(round(w / ratio))))
            return (0, 0, w, crop_h), 0.0

        x0, y0, x1, y1 = bbox
        cx, cy, cw, ch = _expand_bbox_to_ratio(x0, y0, x1, y1, w, h, ratio)
        gray = cv2.cvtColor(analysis_bgr, cv2.COLOR_BGR2GRAY)
        crop_gray = _crop_region_gray(gray, (cx, cy, cw, ch))
        quality = _region_quality(crop_gray)
        conf = _score_saliency(saliency_map, mask, cw, ch, quality)
        return (cx, cy, cw, ch), conf
    except Exception:
        crop_h = max(1, min(h, int(round(w / ratio))))
        return (0, 0, w, crop_h), 0.0


def _get_onnx_face_session() -> Optional[Any]:
    global _ONNX_FACE_SESSION
    if _ONNX_FACE_SESSION is not None:
        return _ONNX_FACE_SESSION
    if ort is None:
        return None

    candidates = []
    env_path = os.environ.get("ANIME_FACE_ONNX_PATH", "").strip()
    if env_path:
        candidates.append(env_path)
    candidates.extend(
        [
            os.path.join("models", "anime_face.onnx"),
            os.path.join("models", "yolov5n-face.onnx"),
            os.path.join("models", "face.onnx"),
        ]
    )

    for path in candidates:
        if not path:
            continue
        if not os.path.exists(path):
            continue
        try:
            _ONNX_FACE_SESSION = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            return _ONNX_FACE_SESSION
        except Exception:
            continue
    return None


def _yolo_like_boxes_from_output(output: np.ndarray, input_w: int, input_h: int) -> List[Tuple[int, int, int, int, float]]:
    boxes: List[Tuple[int, int, int, int, float]] = []
    arr = output
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 5:
        return boxes

    # Case A: xyxy(+score...)
    xyxy_like = float(np.mean((arr[:, 2] > arr[:, 0]) & (arr[:, 3] > arr[:, 1])))
    if xyxy_like > 0.9:
        for row in arr:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            score = float(row[4]) if arr.shape[1] >= 5 else 0.0
            if arr.shape[1] >= 6:
                cls_prob = float(np.max(row[5:]))
                if cls_prob > 0.0:
                    score = score * cls_prob
            if score < 0.25:
                continue
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x1 *= input_w
                x2 *= input_w
                y1 *= input_h
                y2 *= input_h
            boxes.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), score))
        return boxes

    # Case B: cx,cy,w,h,obj,cls...
    for row in arr:
        cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        obj = float(row[4])
        cls_prob = float(np.max(row[5:])) if arr.shape[1] > 5 else 1.0
        score = obj * cls_prob
        if score < 0.25:
            continue
        if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 1.5:
            cx *= input_w
            bw *= input_w
            cy *= input_h
            bh *= input_h
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        boxes.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), score))
    return boxes


def _onnx_face_boxes(analysis_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    session = _get_onnx_face_session()
    if session is None:
        return []
    if cv2 is None:
        return []

    h, w = analysis_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return []

    input_size = 640
    resized = cv2.resize(analysis_bgr, (input_size, input_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))[None, :, :, :]

    try:
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: tensor})
    except Exception:
        return []

    raw_boxes: List[Tuple[int, int, int, int, float]] = []
    for out in outputs:
        raw_boxes.extend(_yolo_like_boxes_from_output(np.asarray(out), input_size, input_size))
    if not raw_boxes:
        return []

    nms_boxes = []
    nms_scores = []
    for x1, y1, x2, y2, score in raw_boxes:
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        nms_boxes.append([x1, y1, bw, bh])
        nms_scores.append(float(score))

    try:
        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, score_threshold=0.25, nms_threshold=0.45)
    except Exception:
        indices = []

    if indices is None or len(indices) == 0:
        selected = list(range(len(raw_boxes)))
    else:
        selected = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in indices]

    sx = float(w) / float(input_size)
    sy = float(h) / float(input_size)

    final_boxes = []
    for idx in selected:
        x1, y1, x2, y2, score = raw_boxes[idx]
        ax1 = int(round(x1 * sx))
        ay1 = int(round(y1 * sy))
        ax2 = int(round(x2 * sx))
        ay2 = int(round(y2 * sy))
        ax1 = max(0, min(ax1, w - 1))
        ay1 = max(0, min(ay1, h - 1))
        ax2 = max(0, min(ax2, w - 1))
        ay2 = max(0, min(ay2, h - 1))
        if ax2 <= ax1 or ay2 <= ay1:
            continue
        final_boxes.append((ax1, ay1, ax2, ay2, float(score)))
    return final_boxes


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

    gray = cv2.cvtColor(analysis_bgr, cv2.COLOR_BGR2GRAY)

    try:
        onnx_faces = _onnx_face_boxes(analysis_bgr)
        if len(onnx_faces) > 0:
            onnx_faces = sorted(
                onnx_faces,
                key=lambda f: (f[4], (f[2] - f[0]) * (f[3] - f[1])),
                reverse=True,
            )
            x1, y1, x2, y2, det_score = onnx_faces[0]
            fw = max(1, x2 - x1)
            fh = max(1, y2 - y1)
            face_aspect = float(fw) / float(max(1, fh))
            face_area_ratio = float(fw * fh) / float(max(1, w * h))
            if face_aspect < 0.5 or face_aspect > 1.8:
                return None, 0.0
            if face_area_ratio < 0.008 or face_area_ratio > 0.45:
                return None, 0.0

            face_region = _crop_region_gray(gray, (x1, y1, fw, fh))
            if not _face_region_valid(face_region):
                return None, 0.0

            cx = x1 + fw / 2.0
            cy = y1 + fh / 2.0
            padded_w = fw * 2.8
            padded_h = fh * 2.9
            target_w, target_h = _fit_ratio_size(padded_w, padded_h, w, h, ratio)
            crop = _crop_from_center(cx, cy, target_w, target_h, w, h)
            crop_gray = _crop_region_gray(gray, crop)
            quality = _region_quality(crop_gray)
            balloon_risk = _balloon_risk(crop_gray)

            conf = _clamp(0.50 * float(det_score) + 0.30 * quality + 0.20 * math.sqrt(max(0.0, face_area_ratio)), 0.0, 1.0)
            conf = conf * (1.0 - 0.65 * balloon_risk)
            if balloon_risk > 0.55:
                conf = conf * 0.35
            conf = _clamp(conf, 0.0, 1.0)
            return crop, conf
    except Exception:
        pass

    cascade = _get_face_cascade()
    if cascade is None:
        return None, 0.0

    try:
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
        face_aspect = float(fw) / float(max(1, fh))
        face_area_ratio = float(fw * fh) / float(max(1, w * h))
        if face_aspect < 0.5 or face_aspect > 1.8:
            return None, 0.0
        if face_area_ratio < 0.008 or face_area_ratio > 0.45:
            return None, 0.0

        face_region = _crop_region_gray(gray, (fx, fy, fw, fh))
        if not _face_region_valid(face_region):
            return None, 0.0

        cx = fx + fw / 2.0
        cy = fy + fh / 2.0

        padded_w = fw * 2.4
        padded_h = fh * 2.6
        target_w, target_h = _fit_ratio_size(padded_w, padded_h, w, h, ratio)
        crop = _crop_from_center(cx, cy, target_w, target_h, w, h)
        crop_gray = _crop_region_gray(gray, crop)
        quality = _region_quality(crop_gray)
        balloon_risk = _balloon_risk(crop_gray)

        conf = _clamp(0.30 + 1.7 * math.sqrt(max(0.0, face_area_ratio)) + 0.20 * quality, 0.0, 1.0)
        conf = conf * (1.0 - 0.65 * balloon_risk)
        if balloon_risk > 0.55:
            conf = conf * 0.35
        conf = _clamp(conf, 0.0, 1.0)
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


def _fallback_crop_with_analysis(
    analysis_bgr: np.ndarray, ratio: float
) -> Tuple[int, int, int, int]:
    h, w = analysis_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return (0, 0, max(1, w), max(1, h))

    target_w = w
    target_h = int(round(target_w / ratio))
    if target_h > h:
        return _fallback_crop(w, h, ratio)

    gray = cv2.cvtColor(analysis_bgr, cv2.COLOR_BGR2GRAY)
    best = None
    best_quality = -1.0

    for top_bias in (0.15, 0.18, 0.20):
        y = int(round(h * top_bias))
        if y + target_h > h:
            y = max(0, h - target_h)
        candidate = (0, y, target_w, target_h)
        region = _crop_region_gray(gray, candidate)
        quality = _region_quality(region)
        if quality > best_quality:
            best_quality = quality
            best = candidate

    if best is None:
        return _fallback_crop(w, h, ratio)
    return best


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


def _select_best_mode(
    sal_crop: Tuple[int, int, int, int],
    sal_conf: float,
    face_crop: Optional[Tuple[int, int, int, int]],
    face_conf: float,
    gray_analysis: np.ndarray,
) -> Tuple[str, float, Tuple[int, int, int, int]]:
    sal_gray = _crop_region_gray(gray_analysis, sal_crop)
    sal_balloon_risk = _balloon_risk(sal_gray)
    sal_effective_conf = _clamp(sal_conf - 0.40 * sal_balloon_risk, 0.0, 1.0)

    face_effective_conf = 0.0
    if face_crop is not None:
        face_gray = _crop_region_gray(gray_analysis, face_crop)
        face_quality = _region_quality(face_gray)
        face_balloon_risk = _balloon_risk(face_gray)
        face_effective_conf = _clamp(0.85 * face_conf + 0.15 * face_quality - 0.25 * face_balloon_risk, 0.0, 1.0)

    use_face = False
    if face_crop is not None:
        if sal_balloon_risk >= BALLOON_RISK_OVERRIDE_THRESHOLD and face_effective_conf >= FACE_OVERRIDE_THRESHOLD:
            use_face = True
        elif face_effective_conf >= FACE_ACCEPT_THRESHOLD and face_effective_conf >= sal_effective_conf:
            use_face = True
        elif sal_effective_conf < SALIENCY_ACCEPT_THRESHOLD and face_effective_conf >= FACE_OVERRIDE_THRESHOLD:
            use_face = True

    if use_face and face_crop is not None:
        return "face", face_effective_conf, face_crop
    if sal_effective_conf >= SALIENCY_ACCEPT_THRESHOLD:
        return "saliency", sal_effective_conf, sal_crop
    return "fallback", 0.0, sal_crop


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
        gray_analysis = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY)
        analysis_fallback = _fallback_crop_with_analysis(analysis, ratio)
        crop = _map_crop_to_original(analysis_fallback, aw, ah, orig_w, orig_h, ratio)
        crop = _ensure_in_bounds(*crop, orig_w, orig_h, ratio)

        sal_crop, sal_conf = _saliency_crop(analysis, ratio)
        mapped_sal_crop = _map_crop_to_original(sal_crop, aw, ah, orig_w, orig_h, ratio)
        mapped_sal_crop = _ensure_in_bounds(*mapped_sal_crop, orig_w, orig_h, ratio)

        face_crop, face_conf = _face_crop(analysis, ratio)
        mapped_face_crop = None
        if face_crop is not None:
            mapped_face_crop = _map_crop_to_original(face_crop, aw, ah, orig_w, orig_h, ratio)
            mapped_face_crop = _ensure_in_bounds(*mapped_face_crop, orig_w, orig_h, ratio)

        mode, effective_conf, mode_crop = _select_best_mode(
            sal_crop=sal_crop,
            sal_conf=sal_conf,
            face_crop=face_crop,
            face_conf=face_conf,
            gray_analysis=gray_analysis,
        )
        if mode == "face" and mapped_face_crop is not None:
            crop = mapped_face_crop
            method = "face"
            confidence = effective_conf
        elif mode == "saliency":
            crop = mapped_sal_crop
            method = "saliency"
            confidence = effective_conf
        elif mode == "fallback":
            # keep deterministic fallback selected earlier
            _ = mode_crop

    crop = _ensure_in_bounds(*crop, orig_w, orig_h, ratio)
    result = _build_result(crop[0], crop[1], crop[2], crop[3], method, confidence)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
