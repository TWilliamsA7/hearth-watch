"""Face templates on the edge (logic only; TensorRT runs on the Jetson).

Run on the laptop: .venv/bin/python -m pytest edge/tests
"""

import io
import json
import os
import sys
from argparse import Namespace
from urllib import error as urlerror

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import run_edge  # noqa: E402
from hearth_edge import face_id as fid  # noqa: E402
from hearth_edge.face import head_box  # noqa: E402

np = pytest.importorskip("numpy")

T = 1791700000.5
REF = dict(fid.REFERENCE)
# A face looking at the camera, eyes 40 px apart: person's right eye on the image left.
KPS = {"right_eye": (300.0, 200.0), "left_eye": (340.0, 200.0), "nose": (320.0, 225.0),
       "left_ear": (355.0, 205.0), "right_ear": (285.0, 205.0)}
HEAD = head_box(KPS, 640, 480)
FACE = (295, 180, 345, 245, 0.97)


def apply(m, point):
    return tuple(m.dot([point[0], point[1], 1.0]))


def test_reference_maps_onto_itself():
    m = fid.similarity_transform(list(REF.values()), list(REF.values()))
    assert np.allclose(m, [[1, 0, 0], [0, 1, 0]], atol=1e-9)


def test_turned_and_shrunk_face_lands_on_the_reference():
    angle, scale, shift = np.radians(35), 0.4, np.array([210.0, 140.0])
    rot = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    seen = [tuple(rot.dot(p) + shift) for p in REF.values()]
    m = fid.similarity_transform(seen, list(REF.values()))
    for got, want in zip(seen, REF.values()):
        assert np.allclose(apply(m, got), want, atol=1e-6)


def test_alignment_never_mirrors_a_face():
    mirrored = [(-x, y) for x, y in REF.values()]
    m = fid.similarity_transform(mirrored, list(REF.values()))
    assert np.linalg.det(m[:, :2]) > 0


def test_align_puts_the_eyes_and_nose_where_sface_expects_them():
    cv2 = pytest.importorskip("cv2")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    marks = {"right_eye": (0, 0, 255), "left_eye": (0, 255, 0), "nose": (255, 0, 0)}
    for name, color in marks.items():
        cv2.circle(image, (int(KPS[name][0]), int(KPS[name][1])), 2, color, -1)
    face = fid.align(image, KPS)
    assert face.shape == (112, 112, 3)
    for name, color in marks.items():
        ys, xs = np.nonzero(np.all(face == color, axis=2))
        assert len(xs), name
        assert abs(xs.mean() - REF[name][0]) < 2 and abs(ys.mean() - REF[name][1]) < 2


def test_face_must_sit_in_this_persons_head_box():
    elsewhere = (500, 50, 560, 120, 0.99)   # someone else's face, more confident
    assert fid.face_in_head([elsewhere, FACE], HEAD) == FACE
    assert fid.face_in_head([elsewhere], HEAD) is None
    assert fid.face_in_head([], HEAD) is None


def test_alignable_needs_both_eyes_and_the_nose_far_enough_apart():
    assert fid.alignable(KPS, 20)
    assert not fid.alignable(KPS, 41)
    for name in ("left_eye", "right_eye", "nose"):
        assert not fid.alignable(dict((k, v) for k, v in KPS.items() if k != name), 20)


class FakeEngine(object):
    def __init__(self, embedding=None):
        self.calls = 0
        self.embedding = embedding or [1.0] + [0.0] * 127

    def embed(self, face):
        assert face.shape == (112, 112, 3)
        self.calls += 1
        return self.embedding


def make_check(boxes, engine=None):
    sent = []
    detect = lambda bgr, roi: (boxes, 180.0)  # noqa: E731
    check = fid.FaceIdCheck(detect, engine or FakeEngine(), sent.append, "jetson-01", "room-01", 20)
    return check, sent


def job(kps=KPS, head=HEAD, person_id="p2"):
    return {"head": head, "kps": kps, "person_id": person_id, "t": T}


BGR = np.zeros((480, 640, 3), dtype=np.uint8)


def test_sface_runs_only_when_a_face_is_in_the_head_box():
    for boxes, kps in (([], KPS), ([(500, 50, 560, 120, 0.99)], KPS),
                       ([FACE], dict((k, v) for k, v in KPS.items() if k != "nose"))):
        engine = FakeEngine()
        check, sent = make_check(boxes, engine)
        assert check(BGR, job(kps)) == (boxes, 180.0)  # presence works exactly as before
        assert engine.calls == 0 and sent == []


def test_no_head_in_view_checks_presence_only():
    engine = FakeEngine()
    check, sent = make_check([FACE], engine)
    boxes, _ = check(BGR, {"head": None, "kps": {}, "person_id": None, "t": T})
    assert boxes == [FACE] and engine.calls == 0 and sent == []


def test_template_fits_the_laptops_edge_route():
    edge_identity = pytest.importorskip("hearth.monitor.edge_identity")
    check, sent = make_check([FACE])
    boxes, ms = check(BGR, job())
    assert ms > 180.0 and len(sent) == 1
    template = sent[0]
    assert template["person_id"] == "p2" and len(template["embedding"]) == 128
    assert "frame" not in template and "image" not in template
    edge_identity.EdgeFaceTemplate(**template)  # the backend's own validation


def test_turns_go_to_whoever_waited_longest():
    turns = fid.FaceTurns()
    people = [("p1", HEAD), ("p2", HEAD), ("p3", HEAD)]
    picks = [turns.pick(people, T + i)[0] for i in range(6)]
    assert picks == ["p1", "p2", "p3", "p1", "p2", "p3"]
    assert turns.pick([("p4", HEAD), ("p2", HEAD)], T + 7)[0] == "p4"   # new person first
    assert turns.pick([], T + 8) is None


def test_face_id_job_takes_people_with_a_head_in_turn():
    turns = fid.FaceTurns()
    frame = {"t": T, "width": 640, "height": 480}
    headless = {"keypoints": {"left_hip": (100.0, 300.0)}, "bbox": (80, 200, 140, 470)}
    people = [("p1", headless, None, None, True), ("p2", {"keypoints": KPS, "bbox": (280, 170, 360, 470)},
                                                   None, None, False)]
    first = run_edge.face_id_job(people, turns, frame, head_box)
    assert first["person_id"] == "p2" and first["head"] == HEAD and first["kps"] is KPS
    nobody = run_edge.face_id_job(people[:1], turns, frame, head_box)
    assert nobody["head"] is None and nobody["person_id"] is None


class Response(object):
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def template(person_id, n=0):
    return {"person_id": person_id, "n": n}


def test_sender_keeps_only_the_newest_template_per_person_and_a_few_people():
    sender = fid.TemplateSender("http://laptop:8000", "tok", max_people=2)
    sender.offer(template("p1", 1))
    sender.offer(template("p1", 2))
    sender.offer(template("p2"))
    sender.offer(template("p3"))
    assert sender.counts["dropped"] == 2
    taken = [sender._take() for _ in range(3)]
    assert taken == [template("p2"), template("p3"), None]


def test_sender_posts_with_the_token_and_remembers_the_answer():
    seen = []

    def opener(req, timeout):
        seen.append(req)
        return Response(b'{"status": "confirming", "matches": 1}')

    sender = fid.TemplateSender("http://laptop:8000/", "tok", opener=opener)
    sender.post(template("p1"))
    assert seen[0].full_url == "http://laptop:8000/face-demo/edge-embeddings"
    assert seen[0].get_header("Authorization") == "Bearer tok"
    assert json.loads(seen[0].data.decode()) == template("p1")
    assert sender.statuses == {"p1": "confirming"} and sender.counts["sent"] == 1


def test_sender_logs_a_refusal_once_and_never_raises():
    logs = []

    def opener(req, timeout):
        raise urlerror.HTTPError(req.full_url, 403, "Forbidden", {},
                                 io.BytesIO(b'{"detail": "Edge face templates need HEARTH_EDGE_FACE_TOKEN"}'))

    sender = fid.TemplateSender("http://laptop:8000", "tok", opener=opener, log=logs.append)
    sender.post(template("p1"))
    sender.post(template("p1"))
    assert sender.counts["failed"] == 2 and len(logs) == 1 and "HEARTH_EDGE_FACE_TOKEN" in logs[0]

    def offline(req, timeout):
        raise urlerror.URLError("connection refused")

    sender = fid.TemplateSender("http://laptop:8000", "tok", opener=offline, log=logs.append)
    sender.post(template("p1"))
    assert sender.counts["failed"] == 1 and len(logs) == 1


def test_read_token(tmp_path):
    path = tmp_path / "token"
    path.write_text("  s3cret \nignored\n")
    assert fid.read_token(str(path)) == "s3cret"
    assert fid.read_token(None, {"HEARTH_EDGE_FACE_TOKEN": "env"}) == "env"
    assert fid.read_token(None, {}) is None


def test_normalized_rejects_useless_vectors():
    assert fid.normalized([3.0, 4.0]) == [0.6, 0.8]
    assert fid.normalized([0.0, 0.0]) is None
    assert fid.normalized([float("nan"), 1.0]) is None
    assert fid.normalized([]) is None


class FakeFace(object):
    model = "res10"

    def detect_bgr(self, bgr, roi):
        return [], 0.0


def args(**changes):
    values = dict(multi_person=True, dry_run=False, face_id_token_file=None, face_id_engine=None,
                  backend="http://laptop:8000", source="jetson-01", room="room-01", face_id_min_eye=20)
    values.update(changes)
    return Namespace(**values)


def test_face_id_stays_off_with_a_reason_rather_than_stopping_monitoring(monkeypatch):
    logs = []
    monkeypatch.setattr(fid, "SFaceEngine", lambda path: pytest.fail("engine loaded"))
    assert run_edge.start_face_id(args(), None, logs.append) == (None, None, None)
    assert run_edge.start_face_id(args(multi_person=False), FakeFace(), logs.append) == (None, None, None)
    monkeypatch.delenv("HEARTH_EDGE_FACE_TOKEN", raising=False)
    assert run_edge.start_face_id(args(), FakeFace(), logs.append) == (None, None, None)
    assert "res10" in logs[0] and "single-person" in logs[1] and "token" in logs[2]


def test_face_id_dry_run_needs_no_token(monkeypatch):
    monkeypatch.delenv("HEARTH_EDGE_FACE_TOKEN", raising=False)

    class Engine(FakeEngine):
        engine_path = "models/face-id/sface_fp16.engine"

    monkeypatch.setattr(fid, "SFaceEngine", lambda path: Engine())
    check, turns, sender = run_edge.start_face_id(args(dry_run=True), FakeFace(), lambda m: None)
    assert isinstance(check, fid.FaceIdCheck) and sender is None


def test_edge_template_names_the_person_on_the_laptop(tmp_path):
    """The edge's own template, sent by its own sender, through the laptop's real route."""
    pytest.importorskip("cryptography")
    from datetime import datetime, timezone
    from cryptography.fernet import Fernet
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from hearth.api.face_demo import create_face_router
    from hearth.monitor.appointments import Appointments
    from hearth.monitor.config import MonitorConfig
    from hearth.monitor.face_demo import FaceDemo
    from hearth.monitor.service import MonitorService
    from hearth_schema import Observation

    now = {"t": T}
    utcnow = lambda: datetime.fromtimestamp(now["t"], timezone.utc)  # noqa: E731

    class Extractor(object):
        def extract(self, frame, **kwargs):
            return [1.0] + [0.0] * 127

    monitor = MonitorService(MonitorConfig(posture_mode="pose_fallback"), clock=lambda: now["t"], utcnow=utcnow)
    faces = FaceDemo(tmp_path / "faces.sqlite", Fernet.generate_key(), Extractor(), clock=lambda: now["t"])
    faces.enroll("Alex", ["a", "b", "c"], True, identify_alerts=True)
    app = FastAPI()
    appointments = Appointments(monitor, ":memory:")
    app.include_router(create_face_router(appointments, monitor, faces, edge_token="tok"))

    with TestClient(app, client=("192.168.55.1", 40000)) as client:
        def opener(req, timeout):
            r = client.post(req.full_url.replace("http://laptop:8000", ""), content=req.data,
                            headers=dict(req.header_items()))
            if r.status_code >= 400:
                raise urlerror.HTTPError(req.full_url, r.status_code, "", {}, io.BytesIO(r.content))
            return Response(r.content)

        sender = fid.TemplateSender("http://laptop:8000", "tok", opener=opener, log=pytest.fail)
        check = fid.FaceIdCheck(lambda bgr, roi: ([FACE], 180.0), FakeEngine(), sender.post,
                                "jetson-01", "room-01", 20)
        upright = dict(torso_angle_deg=5, head_y=.2, hip_y=.5, bbox_aspect=.4)
        floor = dict(torso_angle_deg=90, head_y=.85, hip_y=.85, bbox_aspect=1.8)
        for i in range(12):
            now["t"] = T + i * .5
            monitor.ingest([Observation(obs_id="o%d" % i, timestamp=utcnow(), source="jetson-01",
                                        source_type="edge", room_id="room-01", person_id="p2",
                                        person_detected=True, features=floor if i >= 3 else upright)])
            if i < 3:
                check(BGR, dict(job(), t=now["t"]))
        assert sender.statuses == {"p2": "recognized"} and sender.counts["sent"] == 3
    alert = next(a for a in monitor.alerts.values() if a.kind == "possible_fall")
    assert alert.metadata["resident_label"] == "Alex" and alert.metadata["person_id"] == "p2"
    appointments.db.close()
    faces.db.close()
