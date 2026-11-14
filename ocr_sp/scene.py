"""Text in photos (e.g. chalk numbers on steel billets).

Two stages:
1. EasyOCR's CRAFT detector finds text regions anywhere in a cluttered photo.
2. Each region is cleaned up (dark text on light background, contrast boost)
   and read by TrOCR-large-handwritten, optionally constrained to digits.
"""
from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .recognizer import Result

SCENE_MODEL = "microsoft/trocr-large-handwritten"


def _normalize_crop(crop: Image.Image) -> Image.Image:
    """Grayscale, boost contrast and make the writing dark on a light background."""
    gray = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (mask > 0).mean() < 0.5:  # bright pixels are the minority -> bright text (chalk)
        gray = 255 - gray
    return Image.fromarray(gray).convert("RGB")


class SceneTextReader:
    def __init__(self, model_name: str = SCENE_MODEL, device: str | None = None):
        import easyocr
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = easyocr.Reader(["en"], gpu=self.device == "cuda", verbose=False)
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device).eval()

        tok = self.processor.tokenizer
        self.digit_tokens = [
            i for t, i in tok.get_vocab().items() if tok.convert_tokens_to_string([t]).strip().isdigit()
        ] + [tok.eos_token_id]

    def detect(self, image: np.ndarray, min_size: int = 40) -> list[tuple[int, int, int, int]]:
        horizontal, free = self.detector.detect(image, width_ths=0.1, mag_ratio=1.5, text_threshold=0.5, low_text=0.3)
        boxes = [(x0, y0, x1, y1) for x0, x1, y0, y1 in horizontal[0]]
        for poly in free[0]:
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        return [tuple(int(v) for v in b) for b in boxes if max(b[2] - b[0], b[3] - b[1]) >= min_size]

    @torch.inference_mode()
    def _read(self, crops: list[Image.Image], digits_only: bool) -> list[tuple[str, float]]:
        pixels = self.processor(images=crops, return_tensors="pt").pixel_values.to(self.device)
        kwargs = {"prefix_allowed_tokens_fn": lambda _b, _ids: self.digit_tokens} if digits_only else {}
        out = self.model.generate(
            pixels, max_new_tokens=24, num_beams=4, output_scores=True, return_dict_in_generate=True, **kwargs
        )
        texts = self.processor.batch_decode(out.sequences, skip_special_tokens=True)
        texts = [t.replace(" ", "") if digits_only else t.strip(" .") for t in texts]
        return list(zip(texts, torch.exp(out.sequences_scores).tolist()))

    def recognize(self, image: Image.Image, digits_only: bool = False, min_confidence: float = 0.02) -> Result:
        rgb = image.convert("RGB")
        found = []
        for x0, y0, x1, y1 in self.detect(np.array(rgb)):
            w, h = x1 - x0, y1 - y0
            pad = int(0.15 * min(w, h)) + 4
            crop = rgb.crop((max(0, x0 - pad), max(0, y0 - pad), min(rgb.width, x1 + pad), min(rgb.height, y1 + pad)))
            crop = _normalize_crop(crop)
            # Tall boxes are text written sideways: try both rotations, keep the more confident read.
            candidates = [crop] if w >= h else [crop.rotate(90, expand=True), crop.rotate(-90, expand=True)]
            text, conf = max(self._read(candidates, digits_only), key=lambda r: r[1])
            if text and conf >= min_confidence:
                found.append(((x0, y0, x1, y1), text, conf))
        # Reading order: top to bottom, then left to right.
        found.sort(key=lambda d: (d[0][1] // 60, d[0][0]))

        lines = [text for _, text, _ in found]
        return Result(
            text="\n".join(lines),
            lines=lines,
            confidences=[conf for _, _, conf in found],
            boxes=[box for box, _, _ in found],
        )


def annotate(image: Image.Image, result: Result) -> str:
    """Draw boxes and readings on the image; return it as a base64 JPEG data URL."""
    im = image.convert("RGB").copy()
    im.thumbnail((1280, 1280))
    scale = im.width / image.width
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for (x0, y0, x1, y1), text, conf in zip(result.boxes, result.lines, result.confidences):
        box = [v * scale for v in (x0, y0, x1, y1)]
        color = (40, 200, 70) if conf >= 0.3 else (255, 170, 0)
        draw.rectangle(box, outline=color, width=3)
        label = f"{text} ({conf:.0%})"
        tx, ty = box[0], max(0, box[1] - 26)
        draw.rectangle(draw.textbbox((tx, ty), label, font=font), fill=color)
        draw.text((tx, ty), label, fill="black", font=font)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
