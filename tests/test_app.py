import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ocr_sp import app as app_module
from ocr_sp.recognizer import Result


class FakePage:
    def recognize(self, image, segment=True):
        return Result(text="hello", lines=["hello"])


class FakeScene:
    def recognize(self, image, digits_only=False):
        return Result(text="2519436", lines=["2519436"], confidences=[0.9], boxes=[(1, 1, 30, 15)])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "get_recognizer", lambda: FakePage())
    monkeypatch.setattr(app_module, "get_scene_reader", lambda: FakeScene())
    return TestClient(app_module.app)


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (50, 20), "white").save(buf, "PNG")
    return buf.getvalue()


def test_scene_mode_is_default(client):
    r = client.post("/api/recognize", files=[("files", ("a.png", _png(), "image/png"))])
    assert r.status_code == 200
    assert r.json()["results"][0]["lines"] == ["2519436"]


def test_page_mode(client):
    r = client.post("/api/recognize?mode=page", files=[("files", ("a.png", _png(), "image/png"))])
    assert r.json()["results"][0]["text"] == "hello"


def test_rejects_non_image(client):
    r = client.post("/api/recognize", files=[("files", ("a.txt", b"not an image", "text/plain"))])
    assert r.status_code == 400


def test_scene_result_includes_annotated_image(client):
    r = client.post("/api/recognize", files=[("files", ("a.png", _png(), "image/png"))])
    assert r.json()["results"][0]["annotated"].startswith("data:image/jpeg;base64,")
