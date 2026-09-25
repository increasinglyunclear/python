#!/usr/bin/env python3
"""Join two portrait images side by side with a white gap between them.

By Kevin Walker, Sep 2026, created for https://the-making-of-creativity.com/

Usage:
  python3 side_by_side.py left.jpg right.jpg output.jpg [--gap 10] [--match min|max]
"""
import argparse
from PIL import Image, ImageOps


def resize_to_height(im, height):
    if im.height == height:
        return im
    width = round(im.width * height / im.height)
    return im.resize((width, height), Image.LANCZOS)


def combine(left_path, right_path, out_path, gap=10, match="min"):
    left = ImageOps.exif_transpose(Image.open(left_path)).convert("RGB")
    right = ImageOps.exif_transpose(Image.open(right_path)).convert("RGB")

    target_height = min(left.height, right.height) if match == "min" else max(left.height, right.height)
    left = resize_to_height(left, target_height)
    right = resize_to_height(right, target_height)

    canvas = Image.new("RGB", (left.width + gap + right.width, target_height), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    canvas.save(out_path, quality=95)
    print(f"Saved {out_path} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("output")
    parser.add_argument("--gap", type=int, default=10, help="Gap width in pixels (default: 10)")
    parser.add_argument("--match", choices=["min", "max"], default="min",
                         help="Match heights to the shorter (min, default) or taller (max) image")
    args = parser.parse_args()
    combine(args.left, args.right, args.output, gap=args.gap, match=args.match)
