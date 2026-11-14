"""Command-line entry point: python -m ocr_sp.cli image1.jpg [image2.png ...]"""
from __future__ import annotations

import argparse

from PIL import Image, ImageOps

from .recognizer import DEFAULT_MODEL, HandwritingRecognizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Recognise handwritten text in images.")
    parser.add_argument("images", nargs="+", help="Image files to read")
    parser.add_argument("--mode", choices=["scene", "page"], default="scene",
                        help="scene: text anywhere in a photo (default); page: handwritten page/notes")
    parser.add_argument("--digits", action="store_true", help="scene mode: only read numbers")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="page mode: Hugging Face TrOCR model name")
    parser.add_argument("--single-line", action="store_true", help="page mode: treat each image as one line")
    args = parser.parse_args(argv)

    if args.mode == "scene":
        from .scene import SceneTextReader

        reader = SceneTextReader()
        run = lambda im: reader.recognize(im, digits_only=args.digits)
    else:
        recognizer = HandwritingRecognizer(args.model)
        run = lambda im: recognizer.recognize(im, segment=not args.single_line)

    for path in args.images:
        result = run(ImageOps.exif_transpose(Image.open(path)))
        print(f"=== {path} ===\n{result.text}\n")


if __name__ == "__main__":
    main()
