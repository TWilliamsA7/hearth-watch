"""Recognition is on by default: its key is made once in data/, and failures say what to do."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hearth.api import face_demo
from hearth.monitor.appointments import Appointments


class Extractor:
    def extract(self, frame, **kwargs):
        return [1.0] + [0.0] * 127


def load(tmp_path, **kwargs):
    return face_demo.load_recognition(tmp_path / "faces.sqlite", tmp_path / "face.key",
                                      extractor_factory=Extractor, **kwargs)


def test_turned_off_explicitly(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARTH_FACE_DEMO", "0")
    demo, reason = load(tmp_path)
    assert demo is None and "HEARTH_FACE_DEMO=0" in reason
    assert not (tmp_path / "face.key").exists()


def test_missing_packages_name_the_fix(tmp_path, monkeypatch):
    monkeypatch.delenv("HEARTH_FACE_DEMO", raising=False)
    real_import = __import__

    def no_cv2(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_cv2)
    demo, reason = load(tmp_path)
    assert demo is None and "face-demo" in reason and "start.ps1" in reason


def test_key_is_made_once_and_reused(tmp_path, monkeypatch):
    pytest.importorskip("cryptography")
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    monkeypatch.delenv("HEARTH_FACE_DEMO", raising=False)
    monkeypatch.delenv("HEARTH_FACE_KEY", raising=False)
    demo, reason = load(tmp_path)
    assert reason is None and demo is not None
    key = (tmp_path / "face.key").read_text().strip()
    demo.enroll("Resident A", ["a", "b", "c"], True)
    demo.db.close()
    again, reason = load(tmp_path)
    assert reason is None and len(again.profiles()) == 1
    assert (tmp_path / "face.key").read_text().strip() == key
    again.db.close()


def test_wrong_key_is_explained_without_key_material(tmp_path, monkeypatch):
    pytest.importorskip("cryptography")
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    from cryptography.fernet import Fernet
    monkeypatch.delenv("HEARTH_FACE_DEMO", raising=False)
    monkeypatch.delenv("HEARTH_FACE_KEY", raising=False)
    demo, _ = load(tmp_path)
    demo.enroll("Resident A", ["a", "b", "c"], True)
    demo.db.close()
    other = Fernet.generate_key().decode()
    monkeypatch.setenv("HEARTH_FACE_KEY", other)
    demo, reason = load(tmp_path)
    assert demo is None and "does not match" in reason and other not in reason


def test_status_reports_the_reason(service):
    app = FastAPI()
    app.include_router(face_demo.create_face_router(
        Appointments(service, ":memory:"), service, load=lambda: (None, "Install the face-demo packages.")))
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000)) as client:
        status = client.get("/face-demo/status").json()
    assert status["enabled"] is False and status["message"] == "Install the face-demo packages."
