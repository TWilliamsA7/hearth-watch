import math
from collections import Counter

import pytest
from hearth.sim.floorplan import FloorPlan, to_world
from hearth.sim.generate import ROOMS, GenSpec, _timed, close_calls, generate
from hearth.sim.generator import ResidentTrack
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
    sc = generate(GenSpec(people=2, fall_rate=0.0, collision_rate=0.0), seed)
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


def test_residents_keep_out_of_each_others_way():
    """Priority planning isn't perfect in a crowded room, but brushes are uncommon and brief,
    and actually walking into someone (unplanned) is rare."""
    scenarios = [generate(GenSpec(people=3, room=room, fall_rate=1.0), seed) for room in ROOMS for seed in range(20)]
    calls = [close_calls(sc) for sc in scenarios]
    assert sum(bool(c) for c in calls) <= 0.25 * len(scenarios)  # inside someone's personal space at all
    assert sum(any(g < 0.25 for *_, g in c) for c in calls) <= 0.12 * len(scenarios)  # bodies touching
    seconds_close = sum(len(c) for c in calls) * 0.1
    assert seconds_close <= 0.02 * sum(sc.duration for sc in scenarios)


def test_a_responder_comes_and_kneels_beside_the_faller():
    seen = 0
    for seed in range(20):
        sc = generate(GenSpec(people=3, fall_rate=1.0, recover_rate=0.0, responder_rate=1.0), seed)
        if not sc.tags["responder"]:
            continue
        seen += 1
        assert sc.tags["response_s"] > 0
        faller = next(r for r in sc.cast() if any(s.label == "fall" for s in r.segments))
        fall = next(s for s in faller.segments if s.label == "fall")
        helper = next(r for r in sc.cast() if any(s.tags.get("responder") for s in r.segments))
        assert helper is not faller
        kneel = next(s for s in helper.segments if s.tags.get("responder"))
        assert kneel.label == "bending" and not kneel.alerts
        # Where they knelt: the end of their last walk before kneeling.
        i = helper.segments.index(kneel)
        last_walk = next(s for s in reversed(helper.segments[:i]) if s.x is not None)
        here, there = to_world(sc.scene, last_walk.x, last_walk.depth), to_world(sc.scene, fall.x, fall.depth)
        assert ((here[0] - there[0]) ** 2 + (here[1] - there[1]) ** 2) ** 0.5 < 1.5
    assert seen >= 15


def test_nobody_helps_when_the_faller_gets_up():
    for seed in range(10):
        assert not generate(GenSpec(people=3, fall_rate=1.0, recover_rate=1.0, responder_rate=1.0), seed).tags["responder"]


def test_people_walk_at_an_older_adults_pace_and_turn_before_setting_off():
    for seed in range(10):
        sc = generate(GenSpec(people=2, fall_rate=0.0), seed)
        for r in sc.cast():
            track = ResidentTrack(r, sc.scene)
            walks = {id(p.walk): p.walk for p in track.plans if p.walk is not None}.values()
            assert all(w.cruise <= 1.15 for w in walks if w.length > 0.5)
            # Nobody sets off walking more than ~45 degrees from the way they were facing.
            for i, p in enumerate(track.plans[1:], 1):
                if p.walk is not None and track.plans[i - 1].walk is not p.walk and p.walk.length > 0.3:
                    before = track.heading_at(i - 1, track.plans[i - 1].segment.duration)
                    first = math.atan2(p.walk.points[1][0] - p.walk.points[0][0], p.walk.points[1][1] - p.walk.points[0][1])
                    assert abs(math.remainder(first - before, 2 * math.pi)) <= math.radians(46), (seed, r.resident_id, i)


def test_people_go_to_places_not_random_spots():
    places = {room: [p.at for p in FloorPlan(scene).places()] for room, scene in ROOMS.items()}
    ends, at_place = 0, 0
    for room in ROOMS:
        for seed in range(10):
            sc = generate(GenSpec(people=2, room=room, fall_rate=0.0, collision_rate=0.0), seed)
            for r in sc.cast():
                track = ResidentTrack(r, sc.scene)
                for w in {id(p.walk): p.walk for p in track.plans if p.walk is not None}.values():
                    ends += 1
                    plan = FloorPlan(sc.scene)
                    seat_fronts = [plan.approach(s) for s in plan.seats()]
                    at_place += any(math.dist(w.points[-1], q) < 0.05 for q in places[room] + seat_fronts + list(plan.doors.values()))
    assert at_place >= 0.8 * ends  # the rest: stepping aside, or on the way to a hiding spot


@pytest.mark.parametrize("people", [2, 3])
def test_collisions_are_rare_bumps_that_sometimes_end_in_falls(people):
    results = Counter()
    for seed in range(40):
        sc = generate(GenSpec(people=people, fall_rate=0.0, collision_rate=1.0), seed)
        if sc.tags.get("collision_dropped"):
            results["dropped"] += 1
            assert not sc.tags["collision"] and sc.tags["falls"] == 0
            continue
        assert sc.tags["collision"]
        results[sc.tags["collision_outcome"]] += 1
        bumps = {r.resident_id: next(s for s in r.segments if "bumped" in s.tags) for r in sc.cast()
                 if any("bumped" in s.tags for s in r.segments)}  # fmt: skip
        assert len(bumps) == 2
        (a, sa), (b, sb) = bumps.items()
        assert sa.tags["bumped"] == b and sb.tags["bumped"] == a
        falls = [s for s in bumps.values() if s.label == "fall"]
        assert sc.tags["falls"] == len(falls)
        assert all(s.alerts and s.tags["fault"] == "collision" and s.motion == "collision" for s in falls)
        assert all(not s.alerts and s.tags["hard_negative"] == "bump" for s in bumps.values() if s.label != "fall")
        # They really are at each other's side when it happens.
        times = {rid: next(t for t, s in _timed(r.segments) if s is bumps[rid]) for rid, r in
                 ((r.resident_id, r) for r in sc.cast()) if rid in bumps}  # fmt: skip
        assert abs(times[a] - times[b]) < 0.1
        tracks = {r.resident_id: ResidentTrack(r, sc.scene) for r in sc.cast()}
        assert math.dist(tracks[a].root_at(times[a]), tracks[b].root_at(times[a])) < 0.7
    assert results["dropped"] <= 0.4 * 40
    assert results["both_stumble"] > results["one_falls"] > 0


def test_collisions_only_when_nobody_falls_otherwise():
    for seed in range(10):
        sc = generate(GenSpec(people=3, fall_rate=1.0, collision_rate=1.0), seed)
        assert not sc.tags["collision"] and sc.tags["falls"] == 1
