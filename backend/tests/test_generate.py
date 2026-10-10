import pytest
from hearth.sim.floorplan import FloorPlan, to_world
from hearth.sim.generate import ROOMS, GenSpec, generate
from hearth.sim.runner import run_headless
from hearth.sim.scenario import Scenario, load_scenario, save_scenario

SEEDS = range(40)


def _cases(**spec):
    return [(GenSpec(room=room, people=1 + seed % 4, **spec), seed) for room in ROOMS for seed in SEEDS]


def _truth_is_contiguous(sc: Scenario) -> None:
    truth = sc.ground_truth()
    for r in sc.cast():
        mine = [gt for gt in truth if gt.person_id == r.resident_id]
        assert mine[0].t_start == 0.0 and mine[-1].t_end == pytest.approx(sc.duration)
        assert all(a.t_end == b.t_start for a, b in zip(mine, mine[1:]))


def test_same_seed_same_scenario_and_different_seeds_differ():
    spec = GenSpec(people=3, fall_rate=1.0)
    assert generate(spec, 5) == generate(spec, 5)
    assert generate(spec, 5).residents != generate(spec, 6).residents
    assert generate(spec, 5).scenario_id != generate(GenSpec(people=3, fall_rate=0.9), 5).scenario_id


@pytest.mark.parametrize(("spec", "seed"), _cases(fall_rate=1.0, outage_rate=0.5))
def test_generated_scenarios_are_sound(spec, seed):
    sc = generate(spec, seed)
    assert len(sc.cast()) == spec.people
    assert sc.duration >= spec.duration - 1e-6
    _truth_is_contiguous(sc)

    falls = [s for r in sc.cast() for s in r.segments if s.label == "fall"]
    assert len(falls) == sc.tags["falls"] == 1
    assert falls[0].alerts and falls[0].tags["fault"] == sc.tags["fall_kind"]
    assert 0.0 < falls[0].x < 1.0  # in the camera's view
    assert sum(s.label == "camera_outage" for r in sc.cast() for s in r.segments) <= 1

    # Every walk stays on free floor, apart from stepping into and out of a seat.
    plan = FloorPlan(sc.scene)
    for r in sc.cast():
        here = to_world(sc.scene, r.start_x, r.start_depth)
        for s in r.segments:
            if s.x is None or s.depth is None:
                continue
            there = to_world(sc.scene, s.x, s.depth)
            if s.label == "walking":
                assert plan.line_free(here, there, plan.blockers(here) | plan.blockers(there)), (r.resident_id, s)
            here = there


@pytest.mark.parametrize("seed", SEEDS)
def test_no_falls_when_the_rate_is_zero(seed):
    sc = generate(GenSpec(people=2, fall_rate=0.0), seed)
    assert sc.tags["falls"] == 0
    assert not any(gt.expects_alert for gt in sc.ground_truth())


def test_fall_rate_is_respected_on_average():
    n = sum(generate(GenSpec(fall_rate=0.5), seed).tags["falls"] > 0 for seed in range(200))
    assert 70 <= n <= 130


def test_outages_never_split_a_fall():
    for seed in SEEDS:
        sc = generate(GenSpec(people=2, fall_rate=1.0, outage_rate=1.0), seed)
        for r in sc.cast():
            labels = [s.label for s in r.segments]
            assert labels.count("fall") <= 1
            for a, b in zip(labels, labels[1:]):
                assert not (a == "fall" and b == "camera_outage")


def test_generated_scenarios_run_and_round_trip_through_yaml(tmp_path):
    for seed in range(3):
        sc = generate(GenSpec(people=3, room="bedroom", fall_rate=1.0), seed)
        assert load_scenario(save_scenario(sc, tmp_path / f"{sc.scenario_id}.yaml")) == sc
        run = run_headless(sc)
        assert run.observations and run.ground_truth
