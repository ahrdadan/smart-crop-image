#!/usr/bin/env python3
"""Image conversion helper for job image endpoint."""

from __future__ import annotations

import argparse
import io
import os
import sys

from PIL import Image

try:
    import pillow_avif  # noqa: F401
except Exception:
    pillow_avif = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert image to jpg/avif with optional resize.")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--format", default="jpg", help="Output format: jpg|avif")
    parser.add_argument("--width", type=int, default=0, help="Optional output width")
    parser.add_argument("--quality", type=int, default=95, help="Quality 0..100")
    return parser.parse_args()


def clamp_quality(value: int) -> int:
    return max(0, min(100, int(value)))


def main() -> int:
    args = parse_args()
    source = args.input.strip()
    if not source:
        print("input is required", file=sys.stderr)
        return 1
    if not os.path.isfile(source):
        print(f"input image not found: {source}", file=sys.stderr)
        return 1

    out_format = args.format.strip().lower()
    if out_format in ("jpg", "jpeg"):
        pil_format = "JPEG"
        content_type = "image/jpeg"
    elif out_format == "avif":
        pil_format = "AVIF"
        content_type = "image/avif"
    else:
        print("format must be one of: jpg, avif", file=sys.stderr)
        return 1

    width = max(0, int(args.width))
    quality = clamp_quality(args.quality)

    try:
        img = Image.open(source).convert("RGB")
        if width > 0 and width != img.width:
            target_h = max(1, int(round((img.height * width) / img.width)))
            img = img.resize((width, target_h), Image.Resampling.LANCZOS)

        output = io.BytesIO()
        save_kwargs = {"quality": quality}
        if pil_format == "JPEG":
            save_kwargs["optimize"] = True
        if pil_format == "AVIF":
            save_kwargs["speed"] = 6

        img.save(output, format=pil_format, **save_kwargs)
        sys.stdout.buffer.write(output.getvalue())
        sys.stdout.flush()
        _ = content_type  # keep explicit mapping for readability and future use.
        return 0
    except Exception as err:  # noqa: BLE001
        print(f"image convert failed: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
