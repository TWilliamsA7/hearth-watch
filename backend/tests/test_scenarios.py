import json
from pathlib import Path

import pytest
from hearth.sim.scenario import Scenario, Segment, Variation, load_all, scenario_paths
from hearth_schema.models import GroundTruth
from pydantic import ValidationError

FIXTURES = Path(__file__).resolve().parents[2] / "shared" / "fixtures"


def test_scenario_library_loads():
    scenarios = load_all()
    assert len(scenarios) == len(scenario_paths()) >= 6


@pytest.mark.parametrize("scenario", load_all().values(), ids=lambda s: s.scenario_id)
def test_ground_truth_covers_scenario_contiguously(scenario: Scenario):
    """Every resident's truth runs from 0 to the end of the scenario without gaps."""
    truth = scenario.ground_truth()
    duration = scenario.realize().duration
    assert len(truth) == sum(len(r.segments) for r in scenario.cast())
    assert len({gt.segment_id for gt in truth}) == len(truth)
    for resident in scenario.cast():
        mine = [gt for gt in truth if gt.person_id == resident.resident_id]
        assert mine[0].t_start == 0.0
        assert mine[-1].t_end == pytest.approx(duration)
        for prev, nxt in zip(mine, mine[1:]):
            assert prev.t_end == nxt.t_start


def test_only_falls_expect_alerts_by_default():
    for scenario in load_all().values():
        for gt in scenario.ground_truth():
            assert gt.expects_alert == (gt.truth_label == "fall")


def test_possible_fall_matches_shared_fixture():
    fixture = GroundTruth.model_validate(json.loads((FIXTURES / "ground_truth.example.json").read_text()))
    nominal = load_all()[fixture.scenario_id].model_copy(update={"variation": Variation.none()})
    assert fixture in nominal.ground_truth(seed=fixture.seed)


def test_seed_override_is_recorded_in_ground_truth():
    scenario = load_all()["possible-fall-01"]
    assert {gt.seed for gt in scenario.ground_truth(seed=99)} == {99}


def test_transition_longer_than_segment_is_rejected():
    with pytest.raises(ValidationError):
        Segment(label="sitting", duration=1.0, transition=2.0)


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        Segment(label="sitting", duration=1.0, speed=3)


def test_seeds_vary_the_performance_within_bounds():
    scenario = load_all()["possible-fall-01"]
    variants = [scenario.realize(seed) for seed in range(10)]
    starts = {v.ground_truth()[2].t_start for v in variants}
    assert len(starts) > 5
    for v in variants:
        for nominal, real in zip(scenario.segments, v.segments):
            assert abs(real.duration - nominal.duration) <= nominal.duration * scenario.variation.duration + 1e-3
            assert real.transition <= real.duration


def test_realize_is_deterministic_and_matches_ground_truth():
    scenario = load_all()["walk-and-sit-01"]
    assert scenario.realize(3) == scenario.realize(3)
    assert scenario.ground_truth(3) == scenario.realize(3).ground_truth()
    assert scenario.realize(3).variation == Variation.none()


def test_off_screen_positions_are_not_jittered():
    scenario = load_all()["walk-and-sit-01"]
    for seed in range(5):
        v = scenario.realize(seed)
        assert v.start_x == scenario.start_x == -0.15
        assert v.segments[-2].x == 1.15


def test_single_person_truth_has_no_person_id():
    assert {gt.person_id for gt in load_all()["possible-fall-01"].ground_truth()} == {None}


def test_multi_person_truth_is_per_resident():
    scenario = load_all()["day-room-01"]
    truth = scenario.ground_truth()
    assert {gt.person_id for gt in truth} == {"p1", "p2", "p3", "p4"}
    alerts = [gt for gt in truth if gt.expects_alert]
    assert [(gt.person_id, gt.truth_label) for gt in alerts] == [("p3", "fall")]


def test_residents_vary_independently_per_seed():
    base = load_all()["day-room-01"]
    a, b = base.realize(1), base.realize(2)
    assert a.residents[0].segments != b.residents[0].segments or a.residents[2].segments != b.residents[2].segments
    assert base.realize(1) == base.realize(1)


@pytest.mark.parametrize("bad", [
    {},  # neither
    {"segments": [{"label": "standing", "duration": 1}],
     "residents": [{"resident_id": "p1", "segments": [{"label": "standing", "duration": 1}]}]},  # both
    {"residents": [{"resident_id": "p1", "segments": [{"label": "standing", "duration": 1}]},
                   {"resident_id": "p1", "segments": [{"label": "standing", "duration": 1}]}]},  # duplicate ids
    {"residents": [{"segments": [{"label": "standing", "duration": 1}]}]},  # missing id
])
def test_cast_must_be_unambiguous(bad):
    with pytest.raises(ValidationError):
        Scenario.model_validate({"scenario_id": "x", "title": "x", **bad})
