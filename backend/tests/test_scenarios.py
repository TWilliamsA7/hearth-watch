import json
from pathlib import Path

import pytest
from hearth.sim.scenario import Scenario, Segment, load_all, scenario_paths
from hearth_schema.models import GroundTruth
from pydantic import ValidationError

FIXTURES = Path(__file__).resolve().parents[2] / "shared" / "fixtures"


def test_scenario_library_loads():
    scenarios = load_all()
    assert len(scenarios) == len(scenario_paths()) >= 6


@pytest.mark.parametrize("scenario", load_all().values(), ids=lambda s: s.scenario_id)
def test_ground_truth_covers_scenario_contiguously(scenario: Scenario):
    truth = scenario.ground_truth()
    assert len(truth) == len(scenario.segments)
    assert truth[0].t_start == 0.0
    assert truth[-1].t_end == pytest.approx(scenario.duration)
    for prev, nxt in zip(truth, truth[1:]):
        assert prev.t_end == nxt.t_start
    assert len({gt.segment_id for gt in truth}) == len(truth)


def test_only_falls_expect_alerts_by_default():
    for scenario in load_all().values():
        for gt in scenario.ground_truth():
            assert gt.expects_alert == (gt.truth_label == "fall")


def test_possible_fall_matches_shared_fixture():
    fixture = GroundTruth.model_validate(json.loads((FIXTURES / "ground_truth.example.json").read_text()))
    truth = load_all()[fixture.scenario_id].ground_truth(seed=fixture.seed)
    assert fixture in truth


def test_seed_override_is_recorded_in_ground_truth():
    scenario = load_all()["possible-fall-01"]
    assert {gt.seed for gt in scenario.ground_truth(seed=99)} == {99}


def test_transition_longer_than_segment_is_rejected():
    with pytest.raises(ValidationError):
        Segment(label="sitting", duration=1.0, transition=2.0)


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        Segment(label="sitting", duration=1.0, speed=3)
