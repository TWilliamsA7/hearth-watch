"""Preview panel text and backend-view polling tests. Run: .venv/bin/python -m pytest edge/tests"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hearth_edge import backend_view as bv  # noqa: E402
from hearth_edge import contract as c  # noqa: E402
from hearth_edge import features as f  # noqa: E402
from hearth_edge import overlay as ov  # noqa: E402

NOW = 1791700000.0


# --- text lines -------------------------------------------------------------------------

def test_status_line():
    line = ov.status_line(13.62, 10, "backend online", "04:58:12")
    assert line.startswith("13.6 FPS | 10 obs/s | backend online") and line.endswith("04:58:12")
    assert ov.status_line(None, 0, "DRY RUN (nothing sent)", "x").startswith("- FPS | 0 obs/s | DRY RUN")


def test_person_line_with_and_without_face_result():
    assert ov.person_line(True, 12, (True, 0.991, 0.34)) == \
        "person: yes (12 keypoints) | face: yes 0.99 (0.3 s ago)"
    assert ov.person_line(True, 5, (False, None, 1.2)) == "person: yes (5 keypoints) | face: no (1.2 s ago)"
    assert ov.person_line(False, 0, None) == "person: no | face: -"


def test_features_line_shows_dash_for_unmeasured():
    line = ov.features_line({"torso_angle_deg": 4.2, "hip_y": 0.551, "vertical_velocity": 0.012})
    assert line == "torso 4 deg | head - | hip 0.55 | box - | drop +0.01/s"
    assert ov.features_line(None).count("-") == 5


def test_backend_lines_for_each_state():
    assert ov.backend_lines(None, "jetson-01") == []  # dry run: no backend view
    assert ov.backend_lines({"reachable": None}, "jetson-01")[0][0] == "backend view: waiting..."
    text, colour = ov.backend_lines({"reachable": False, "error": "HTTP 404"}, "jetson-01")[0]
    assert text == "backend view: unavailable (HTTP 404)" and colour == ov.GREY
    assert "no status for jetson-01" in ov.backend_lines({"reachable": True, "source": None}, "jetson-01")[0][0]

    standing = {"reachable": True, "alerts": [],
                "source": {"online": True, "posture": "standing", "stale": False}}
    assert ov.backend_lines(standing, "jetson-01") == [
        ("backend: online | posture standing", ov.GREEN), ("alerts: none open", ov.GREY)]

    fallen = {"reachable": True,
              "source": {"online": True, "posture": "fallen", "stale": False},
              "alerts": [("possible_fall", "open", 4.2, None, None),
                         ("possible_fall", "acknowledged", 30.0, "Sam", None)]}
    lines = ov.backend_lines(fallen, "jetson-01")
    assert lines[0] == ("backend: online | posture fallen", ov.RED)
    assert lines[1] == ("ALERT possible fall: open 4 s", ov.RED)
    assert lines[2] == ("ALERT possible fall: acknowledged 30 s (Sam)", ov.AMBER)

    offline = {"reachable": True, "alerts": [], "source": {"online": False, "posture": "unknown", "stale": True}}
    assert ov.backend_lines(offline, "jetson-01")[0] == \
        ("backend: OFFLINE | posture unknown (stale data)", ov.AMBER)


def test_head_point_matches_head_y():
    kps = {"nose": (320, 60), "left_eye": (310, 50), "right_eye": (330, 50)}
    assert f.head_point(kps) == pytest.approx((320.0, 53.333), abs=0.01)
    assert f.head_y(kps, 480) == pytest.approx(53.333 / 480, abs=1e-4)
    assert f.head_point({"left_hip": (1, 2)}) is None


# --- summaries of the backend's JSON ----------------------------------------------------

def test_source_summary_picks_this_source():
    sources = [{"source": "sim", "online": True, "posture": "fallen"},
               {"source": "jetson-01", "online": True, "posture": "standing", "observation_stale": False}]
    assert bv.source_summary(sources, "jetson-01") == {"online": True, "posture": "standing", "stale": False}
    assert bv.source_summary(sources, "jetson-02") is None


def test_recovered_alert_stops_counting_and_says_review():
    view = {"reachable": True, "source": {"online": True, "posture": "standing", "stale": False},
            "alerts": [("possible_fall", "open", 120.0, None, 45.0)]}
    assert ov.backend_lines(view, "jetson-01")[1] == \
        ("ALERT possible fall: open, up again after 45 s - review on /nurses", ov.AMBER)
    off = dict(view, alerts=[("source_offline", "open", 60.0, None, 8.0)])
    assert "reconnected after 8 s" in ov.backend_lines(off, "jetson-01")[1][0]


def test_alert_summary_skips_resolved_and_sorts_newest_first():
    alerts = [
        {"kind": "possible_fall", "status": "resolved", "opened_at": c.utc_iso(NOW - 5)},
        {"kind": "possible_fall", "status": "acknowledged", "opened_at": c.utc_iso(NOW - 60),
         "accepted_by_name": "Sam"},
        {"kind": "help_requested", "status": "open", "opened_at": c.utc_iso(NOW - 3)},
    ]
    out = bv.alert_summary(alerts, NOW)
    assert [(k, s) for k, s, _, _, _ in out] == [("help_requested", "open"), ("possible_fall", "acknowledged")]
    assert out[0][2] == pytest.approx(3.0, abs=0.01) and out[1][3] == "Sam"
    assert out[0][4] is None  # not recovered yet


def test_alert_summary_reports_time_down_once_recovered():
    a = {"kind": "possible_fall", "status": "open", "opened_at": c.utc_iso(NOW - 120),
         "recovered_at": c.utc_iso(NOW - 75)}
    assert bv.alert_summary([a], NOW)[0][4] == pytest.approx(45.0, abs=0.01)


# --- polling a backend over HTTP ----------------------------------------------------------

class FakeTeamBackend(object):
    def __init__(self, sources, alerts, has_pages=True):
        seen = self.paths = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append(self.path)
                if not has_pages:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = sources if self.path == "/sources" else alerts
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_backend_view_reads_posture_and_alerts():
    backend = FakeTeamBackend(
        [{"source": "jetson-01", "online": True, "posture": "fallen", "observation_stale": False}],
        [{"kind": "possible_fall", "status": "open", "opened_at": c.utc_iso(NOW - 2)}])
    try:
        view = bv.BackendView(backend.url, "jetson-01", clock=lambda: NOW)
        state = view.poll_once()
    finally:
        backend.close()
    assert state["reachable"] and state["source"]["posture"] == "fallen"
    assert state["alerts"][0][:2] == ("possible_fall", "open")
    assert "/alerts?source=jetson-01" in backend.paths  # only this camera's alerts
    assert view.snapshot() == state


def test_backend_view_without_those_pages_is_unavailable_not_an_error():
    backend = FakeTeamBackend([], [], has_pages=False)  # like the stand-in stub_backend.py
    try:
        state = bv.BackendView(backend.url, "jetson-01").poll_once()
    finally:
        backend.close()
    assert state == {"reachable": False, "error": "HTTP 404"}


def test_backend_view_unreachable():
    state = bv.BackendView("http://127.0.0.1:1", "jetson-01", timeout_s=0.3).poll_once()
    assert state["reachable"] is False


def test_person_line_shows_held_keypoints():
    assert ov.person_line(True, 12, None, n_held=2) == "person: yes (12 keypoints, 2 held) | face: -"
    assert ov.person_line(True, 12, None, n_held=0) == "person: yes (12 keypoints) | face: -"


def test_person_line_shows_rotated_pose_source():
    assert ov.person_line(True, 13, None, source="rotated CCW") == \
        "person: yes (13 keypoints, rotated CCW) | face: -"
