# OCR_SP: Handwriting Recognition

Reads handwritten text and numbers from uploaded images. Two modes:

- **scene** (default): text anywhere in a photo, such as chalk numbers on
  steel billets. Uses [EasyOCR](https://github.com/JaidedAI/EasyOCR) to detect
  and read text regions. Use the digits-only option for ID numbers.
- **page**: handwritten pages or notes. Each page is split into lines with
  OpenCV, then read by
  [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

The first run of each mode downloads its model (EasyOCR ~100 MB, TrOCR ~1.3 GB).

## Web app

```bash
uvicorn ocr_sp.app:app --reload
```

Open http://127.0.0.1:8000, drop in one or more images and click **Recognise**.

API: `POST /api/recognize` (multipart field `files`, optional query
`mode=scene|page`, `digits_only=true`, `single_line=true`) returns `{"results": [{"filename", "text", "lines"}]}`.

## CLI

```bash
python -m ocr_sp.cli billets.jpg --digits
python -m ocr_sp.cli notes.jpg --mode page
python -m ocr_sp.cli word.png --mode page --single-line
python -m ocr_sp.cli page.jpg --mode page --model microsoft/trocr-large-handwritten
```

## Tests

```bash
pytest
```

## Tips

- Best results come from evenly lit scans with dark ink on a plain background.
- Line segmentation assumes roughly horizontal lines. For a crop that is
  already a single line, use `--single-line` (or tick the box in the web UI).
- `trocr-large-handwritten` is more accurate but slower.
- The models are trained on English handwriting.

## License

MIT, see [LICENSE](LICENSE).
