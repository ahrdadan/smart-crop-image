#!/usr/bin/env python3
"""Lightweight fallback thumbnail worker (no YOLO dependency)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from PIL import Image, ImageOps

DEFAULT_GAP = 5
DEFAULT_WIDTH = 1200
TARGET_PORTRAIT_RATIO = 3 / 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fallback chapter thumbnail worker")
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


def clean_image_paths(paths: list[Any]) -> list[str]:
    out: list[str] = []
    for path in paths:
        text = str(path).strip()
        if not text:
            continue
        if os.path.isfile(text):
            out.append(text)
    return out


def portrait_crop_bbox(width: int, height: int, target_ratio: float = TARGET_PORTRAIT_RATIO) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        return 0, 0, 1, 1

    ratio = width / float(height)
    if ratio > target_ratio:
        crop_w = max(1, int(round(height * target_ratio)))
        x0 = max(0, (width - crop_w) // 2)
        x1 = min(width, x0 + crop_w)
        return x0, 0, x1, height

    crop_h = max(1, int(round(width / target_ratio)))
    y0 = max(0, (height - crop_h) // 2)
    y1 = min(height, y0 + crop_h)
    return 0, y0, width, y1


def candidate_score(width: int, height: int) -> float:
    ratio = width / float(max(1, height))
    closeness = 1.0 - min(1.0, abs(ratio - TARGET_PORTRAIT_RATIO) / TARGET_PORTRAIT_RATIO)
    portrait_bonus = 1.0 if height >= width else 0.35
    area = min(1.0, (width * height) / float(1600 * 2400))
    return (0.55 * portrait_bonus) + (0.35 * closeness) + (0.10 * area)


def cover_resize(img: Image.Image, target_w: int, target_h: int, centering=(0.5, 0.5)) -> Image.Image:
    return ImageOps.fit(img.convert("RGB"), (target_w, target_h), method=Image.LANCZOS, centering=centering)


def merge_two_images_to_16x9(
    left_img: Image.Image,
    right_img: Image.Image,
    out_path: str,
    width: int,
    gap: int,
    background: tuple[int, int, int] = (0, 0, 0),
) -> tuple[str, tuple[int, int]]:
    out_w = max(320, int(width))
    out_h = int(round(out_w * 9 / 16))
    slot_w = max(1, (out_w - max(0, gap)) // 2)
    slot_h = out_h

    left = cover_resize(left_img, slot_w, slot_h, centering=(0.5, 0.4))
    right = cover_resize(right_img, slot_w, slot_h, centering=(0.5, 0.4))

    canvas = Image.new("RGB", (out_w, out_h), background)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (slot_w + max(0, gap), 0))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path, quality=95)
    return out_path, (out_w, out_h)


def run(payload: dict[str, Any]) -> dict[str, Any]:
    image_paths = clean_image_paths(payload.get("image_paths") or [])
    if not image_paths:
        raise RuntimeError("image_paths kosong atau tidak valid.")

    out_path = str(payload.get("output_path") or "").strip()
    if not out_path:
        raise RuntimeError("output_path wajib diisi untuk fallback composer.")

    width = max(320, to_int(payload.get("width"), DEFAULT_WIDTH))
    gap = max(0, to_int(payload.get("gap"), DEFAULT_GAP))
    fallback_reason = str(payload.get("fallback_reason") or "").strip()

    candidates: list[dict[str, Any]] = []
    for idx, path in enumerate(image_paths):
        with Image.open(path) as img:
            pil = img.convert("RGB")
            w, h = pil.size
            x0, y0, x1, y1 = portrait_crop_bbox(w, h)
            crop = pil.crop((x0, y0, x1, y1))
            candidates.append(
                {
                    "idx": idx,
                    "path": path,
                    "total": float(candidate_score(w, h)),
                    "bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "crop": crop,
                }
            )

    if not candidates:
        raise RuntimeError("Tidak ada gambar yang bisa diproses.")

    ranked = sorted(candidates, key=lambda item: item["total"], reverse=True)
    top2 = ranked[:2]
    if len(top2) == 1:
        top2 = [top2[0], top2[0]]

    out_file, size = merge_two_images_to_16x9(
        left_img=top2[0]["crop"],
        right_img=top2[1]["crop"],
        out_path=out_path,
        width=width,
        gap=gap,
    )

    return {
        "out_path": out_file,
        "size": [int(size[0]), int(size[1])],
        "picked": [
            {
                "idx": int(top2[0]["idx"]),
                "path": top2[0]["path"],
                "total": float(top2[0]["total"]),
                "text_ratio": 0.0,
                "in_edge": False,
                "bbox": top2[0]["bbox"],
            },
            {
                "idx": int(top2[1]["idx"]),
                "path": top2[1]["path"],
                "total": float(top2[1]["total"]),
                "text_ratio": 0.0,
                "in_edge": False,
                "bbox": top2[1]["bbox"],
            },
        ],
        "fallback_used": True,
        "fallback_reason": fallback_reason or "primary smart worker failed",
    }


def main() -> int:
    try:
        payload = parse_payload(parse_args())
        info = run(payload)
        print(json.dumps(info, separators=(",", ":")))
        return 0
    except Exception as err:  # noqa: BLE001
        print(f"fallback_thumb failed: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
