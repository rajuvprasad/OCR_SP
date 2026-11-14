"""Handwriting recognition with Microsoft's TrOCR model."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .segment import segment_lines

DEFAULT_MODEL = "microsoft/trocr-base-handwritten"


@dataclass
class Result:
    text: str
    lines: list[str]
    confidences: list[float] = field(default_factory=list)
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)


class HandwritingRecognizer:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, batch_size: int = 8):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def recognize_lines(self, lines: list[Image.Image]) -> list[str]:
        texts: list[str] = []
        for i in range(0, len(lines), self.batch_size):
            batch = [im.convert("RGB") for im in lines[i : i + self.batch_size]]
            pixels = self.processor(images=batch, return_tensors="pt").pixel_values.to(self.device)
            ids = self.model.generate(pixels, max_new_tokens=128)
            texts.extend(t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True))
        return texts

    def recognize(self, image: Image.Image, segment: bool = True) -> Result:
        crops = segment_lines(image) if segment else [image]
        lines = [t for t in self.recognize_lines(crops) if t]
        return Result(text="\n".join(lines), lines=lines)
