from datetime import timedelta

import pytest
from hearth.sim.generator import SIM_EPOCH, NoiseConfig, ObservationGenerator
from hearth.sim.scenario import load_all
from hearth_schema.models import Heartbeat, Observation

SCENARIOS = load_all()
ALL = pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=lambda s: s.scenario_id)


def dump(gen):
    return [item.model_dump_json() for item in gen.stream()]


def seconds(item):
    return (item.timestamp - SIM_EPOCH).total_seconds()


def observations(scenario, noise=None, seed=None):
    gen = ObservationGenerator(scenario, seed, noise=noise)
    return [o for o in gen.stream() if isinstance(o, Observation)]


def in_segment(obs, scenario, label, skip=0.0):
    gt = next(g for g in scenario.ground_truth() if g.truth_label == label)
    return [o for o in obs if gt.t_start + skip <= seconds(o) < gt.t_end]


@ALL
def test_same_seed_reproduces_the_run(scenario):
    assert dump(ObservationGenerator(scenario)) == dump(ObservationGenerator(scenario))


@ALL
def test_different_seed_changes_the_run(scenario):
    assert dump(ObservationGenerator(scenario, seed=1)) != dump(ObservationGenerator(scenario, seed=2))


@ALL
def test_stream_is_ordered_labelled_simulation(scenario):
    items = list(ObservationGenerator(scenario).stream())
    times = [seconds(i) for i in items]
    assert times == sorted(times)
    assert all(i.source_type == "simulation" and i.room_id == scenario.room_id for i in items)
    obs = [i for i in items if isinstance(i, Observation)]
    assert all(o.scenario_id == scenario.scenario_id for o in obs)
    assert len({o.obs_id for o in obs}) == len(obs)


@ALL
def test_ground_truth_never_enters_the_stream(scenario):
    for line in dump(ObservationGenerator(scenario)):
        assert "truth_label" not in line and "expects_alert" not in line


@ALL
def test_figure_moves_continuously(scenario):
    """The visual feed must never teleport: bounded joint movement between frames."""
    gen = ObservationGenerator(scenario, noise=NoiseConfig.off())
    prev = None
    for _, t, i in gen.frame_times():
        pose = gen.pose_at(i, t - gen.scenario.segment_times()[i][0])
        if prev is not None:
            jump = max(abs(pose[j][0] - prev[j][0]) + abs(pose[j][1] - prev[j][1]) for j in pose)
            assert jump < 0.2, f"t={t:.2f}s joint jump {jump:.3f}"
        prev = pose


def test_camera_outage_is_silent():
    sc = SCENARIOS["camera-outage-01"]
    gt = next(g for g in sc.ground_truth() if g.truth_label == "camera_outage")
    items = list(ObservationGenerator(sc).stream())
    assert not [i for i in items if gt.t_start <= seconds(i) < gt.t_end]
    beats = [seconds(i) for i in items if isinstance(i, Heartbeat)]
    gaps = [b - a for a, b in zip(beats, beats[1:])]
    assert max(gaps) >= gt.t_end - gt.t_start
    assert sorted(gaps)[len(gaps) // 2] == pytest.approx(1.0)


def test_absent_means_no_person():
    sc = SCENARIOS["walk-and-sit-01"]
    absent = in_segment(observations(sc), sc, "absent")
    assert absent and not any(o.person_detected or o.keypoints for o in absent)


def test_occlusion_hides_lower_body():
    sc = SCENARIOS["occlusion-01"]
    occluded = in_segment(observations(sc, NoiseConfig.off()), sc, "occluded", skip=0.5)
    names = {k.name for o in occluded for k in o.keypoints or []}
    assert occluded and "nose" in names
    assert not names & {"left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"}


def test_postures_produce_expected_features():
    off = NoiseConfig.off()
    fall_sc, bend_sc = SCENARIOS["possible-fall-01"], SCENARIOS["bend-pickup-01"]
    fall_obs, bend_obs = observations(fall_sc, off), observations(bend_sc, off)

    standing = in_segment(fall_obs, fall_sc, "standing", skip=1.0)
    assert all(o.features.torso_angle_deg < 10 for o in standing)
    on_floor = in_segment(fall_obs, fall_sc, "fall", skip=2.0)
    assert all(o.features.torso_angle_deg > 70 for o in on_floor)
    assert min(o.features.hip_y for o in on_floor) > max(o.features.hip_y for o in standing) + 0.15

    bending = in_segment(bend_obs, bend_sc, "bending", skip=1.0)
    assert all(o.features.torso_angle_deg > 45 for o in bending)
    assert all(abs(o.features.hip_y - standing[0].features.hip_y) < 0.05 for o in bending)


def test_fall_descends_faster_than_lying_down():
    off = NoiseConfig.off()

    def peak_velocity(sc, label):
        return max(o.features.vertical_velocity or 0.0 for o in in_segment(observations(sc, off), sc, label))

    fall = peak_velocity(SCENARIOS["possible-fall-01"], "fall")
    lying = peak_velocity(SCENARIOS["lying-intentional-01"], "lying_intentional")
    assert fall > 2 * lying


def test_custom_start_time_shifts_timestamps_only():
    sc = SCENARIOS["possible-fall-01"]
    later = SIM_EPOCH + timedelta(hours=3)
    a = list(ObservationGenerator(sc).stream())
    b = list(ObservationGenerator(sc, start=later).stream())
    assert [i.timestamp + timedelta(hours=3) for i in a] == [i.timestamp for i in b]
