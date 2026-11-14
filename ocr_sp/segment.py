"""Split a page image into individual text-line crops.

TrOCR recognises a single line at a time, so full pages are first segmented
using a horizontal projection profile of the binarised ink.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, ink = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs where mask is True."""
    runs, start = [], None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def segment_lines(
    image: Image.Image,
    min_line_height: int = 10,
    merge_gap: int = 6,
    pad: int = 8,
) -> list[Image.Image]:
    """Return line crops (top to bottom). Falls back to the whole image."""
    rgb = image.convert("RGB")
    gray = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
    ink = _binarize(gray)

    # Smear ink horizontally so words on the same line join up.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, gray.shape[1] // 30), 1))
    smeared = cv2.dilate(ink, kernel)

    profile = smeared.sum(axis=1) / 255
    rows = _runs(profile > max(2, 0.01 * gray.shape[1]))

    merged: list[tuple[int, int]] = []
    for start, end in rows:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    h, w = gray.shape
    crops = []
    for top, bottom in merged:
        if bottom - top < min_line_height:
            continue
        cols = np.where(ink[top:bottom].sum(axis=0) > 0)[0]
        if cols.size == 0:
            continue
        box = (
            max(0, int(cols[0]) - pad),
            max(0, top - pad),
            min(w, int(cols[-1]) + 1 + pad),
            min(h, bottom + pad),
        )
        crops.append(rgb.crop(box))

    return crops or [rgb]
