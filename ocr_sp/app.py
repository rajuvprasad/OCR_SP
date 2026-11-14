"""Web app: upload handwritten images and get the recognised text back.

Run with:  uvicorn ocr_sp.app:app --reload
"""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from .recognizer import HandwritingRecognizer
from .scene import SceneTextReader, annotate

STATIC = Path(__file__).parent / "static"
MAX_BYTES = 15 * 1024 * 1024

app = FastAPI(title="Handwriting OCR")


@lru_cache(maxsize=1)
def get_recognizer() -> HandwritingRecognizer:
    return HandwritingRecognizer()


@lru_cache(maxsize=1)
def get_scene_reader() -> SceneTextReader:
    return SceneTextReader()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/api/recognize")
async def recognize(
    files: list[UploadFile] = File(...),
    mode: Literal["scene", "page"] = "scene",
    digits_only: bool = False,
    single_line: bool = False,
) -> dict:
    results = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"{f.filename} is larger than 15 MB")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except UnidentifiedImageError:
            raise HTTPException(400, f"{f.filename} is not a readable image")
        image = ImageOps.exif_transpose(image)  # phone photos are often stored rotated

        if mode == "scene":
            result = get_scene_reader().recognize(image, digits_only=digits_only)
        else:
            result = get_recognizer().recognize(image, segment=not single_line)
        results.append({
            "filename": f.filename,
            "text": result.text,
            "lines": result.lines,
            "confidences": result.confidences,
            "annotated": annotate(image, result) if result.boxes else None,
        })
    return {"results": results}
