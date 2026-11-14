import cv2
import numpy as np
from PIL import Image

from ocr_sp.segment import segment_lines


def _page(lines: int) -> Image.Image:
    img = np.full((60 * lines + 40, 600, 3), 255, np.uint8)
    for i in range(lines):
        cv2.putText(img, f"line number {i}", (20, 50 + 60 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    return Image.fromarray(img)


def test_splits_into_lines():
    assert len(segment_lines(_page(3))) == 3


def test_blank_image_falls_back_to_whole():
    blank = Image.new("RGB", (200, 100), "white")
    crops = segment_lines(blank)
    assert len(crops) == 1 and crops[0].size == (200, 100)


def test_normalize_crop_makes_chalk_dark():
    from ocr_sp.scene import _normalize_crop

    img = np.full((60, 300, 3), 40, np.uint8)  # dark steel
    cv2.putText(img, "2519436", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (230, 230, 230), 3)  # white chalk
    out = np.array(_normalize_crop(Image.fromarray(img)).convert("L"))
    assert out.mean() > 128  # background is now light
