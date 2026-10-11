"""Following everyone in view (hearth_edge/people.py) and their per-person observations.

Run on the laptop: .venv/bin/python -m pytest edge/tests
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import run_edge  # noqa: E402
from hearth_edge import contract  # noqa: E402
from hearth_edge.people import PeopleTracker  # noqa: E402

hs = pytest.importorskip("hearth_schema")

FRAME = {"t": 1791700000.5, "width": 640, "height": 480}


def pose_at(x, y=250):
    return {"keypoints": {"nose": (x, y - 190), "neck": (x, y - 140), "left_hip": (x - 15, y),
                          "right_hip": (x + 15, y)},
            "bbox": (x - 40, y - 200, x + 40, y + 195)}


def ids_of(assigned):
    return [person.person_id for person, _ in assigned]


def test_each_person_keeps_their_id_as_they_move():
    tr = PeopleTracker(max_dist=160)
    assert ids_of(tr.assign([pose_at(100), pose_at(500)], 0.0)) == ["p1", "p2"]
    # both walk a little; the order they're given in doesn't matter
    assert ids_of(tr.assign([pose_at(530), pose_at(120)], 0.1)) == ["p2", "p1"]


def test_someone_walking_in_gets_a_new_id_and_ids_are_not_reused():
    tr = PeopleTracker(max_dist=160, keep_s=1.0)
    tr.assign([pose_at(100)], 0.0)
    assert ids_of(tr.assign([pose_at(100), pose_at(500)], 0.1)) == ["p1", "p2"]
    tr.assign([pose_at(100)], 0.6)
    tr.assign([pose_at(100)], 1.2)
    assert [p.person_id for p in tr.ended(1.2)] == ["p2"]
    assert ids_of(tr.assign([pose_at(100), pose_at(500)], 1.3)) == ["p1", "p3"]


def test_brief_dropout_is_missing_not_ended():
    tr = PeopleTracker(max_dist=160, keep_s=1.0)
    tr.assign([pose_at(100)], 0.0)
    # poseNet loses someone for 0.8 s: still the same person when they come back
    assert [p.person_id for p in tr.missing(0.8)] == ["p1"] and tr.ended(0.8) == []
    assert ids_of(tr.assign([pose_at(110)], 0.85)) == ["p1"]
    assert [p.person_id for p in tr.ended(2.0)] == ["p1"]
    assert tr.ended(2.1) == []  # reported once


def test_people_walking_side_by_side_keep_their_ids():
    tr = PeopleTracker(max_dist=160)
    tr.assign([pose_at(200), pose_at(300)], 0.0)
    # p1 moved right to 260, p2 to 360. Closest-pair-first would give 260 to p2 (40 px away);
    # the smallest total movement keeps both
    assert ids_of(tr.assign([pose_at(260), pose_at(360)], 0.1)) == ["p1", "p2"]


def test_each_person_has_their_own_velocity():
    tr = PeopleTracker(max_dist=160)
    a, b = tr.assign([pose_at(100), pose_at(500)], 0.0)
    assert a[0].velocity is not b[0].velocity and a[0].hold is not b[0].hold


def prediction(label):
    return {"model_id": "hearth/posture-mlp", "model_revision": "abc", "input_processing": "original",
            "label": label, "confidence": 0.9, "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.9},
            "detection_count": 1, "uncertainty_reason": None}


def test_one_observation_per_person_passes_the_contract():
    ids = contract.ObservationIds("jetson-01", start=1791700000)
    people = [("p1", pose_at(100), None, prediction("standing"), True),
              ("p2", pose_at(500), None, prediction("fallen"), False)]
    out = run_edge.observations_for_people(ids, FRAME, people, ["p7"], "jetson-01", "room-01",
                                           face_present=True, face_confidence=0.8)
    models = [hs.Observation.model_validate(o) for o in out]
    assert [(m.person_id, m.person_detected) for m in models] == [("p1", True), ("p2", True), ("p7", False)]
    assert [m.posture_prediction.label if m.posture_prediction else None for m in models] == [
        "standing", "fallen", None]
    assert models[0].face_present is True and models[1].face_present is None  # face goes on the followed one
    assert len({m.obs_id for m in models}) == 3 and len({m.timestamp for m in models}) == 1


def test_nobody_in_view_sends_one_observation_without_a_person_id():
    ids = contract.ObservationIds("jetson-01", start=1791700000)
    out = run_edge.observations_for_people(ids, FRAME, [], [], "jetson-01", "room-01")
    assert len(out) == 1 and "person_id" not in out[0] and not out[0]["person_detected"]
    hs.Observation.model_validate(out[0])
    # someone only briefly missing: skip, don't say "nobody"
    assert run_edge.observations_for_people(ids, FRAME, [], [], "jetson-01", "room-01",
                                            briefly_missing=True) == []


def test_single_person_flag():
    assert run_edge.parse_args([]).multi_person
    assert not run_edge.parse_args(["--single-person"]).multi_person
