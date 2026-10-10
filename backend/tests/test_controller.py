import asyncio
from datetime import UTC, datetime, timedelta

import pytest
import stub_monitor
from hearth.eval.batch import PerPersonMonitor
from hearth.sim.controller import SimController
from hearth.sim.scenario import load_all
from hearth_schema.models import GroundTruth, Heartbeat, Observation

SCENARIOS = load_all()
DT = 0.05


class FakeClock:
    def __init__(self):
        self.t = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)

    def now(self):
        return self.t

    async def sleep(self, seconds):
        self.t += timedelta(seconds=seconds)
        await asyncio.sleep(0)


class Pipeline:
    """Stand-in for the monitoring core's ingest(): records what arrives, runs a test-double monitor."""

    def __init__(self):
        self.monitor = PerPersonMonitor(stub_monitor.create_monitor)
        self.received = []

    async def ingest(self, items):
        self.received.extend(items)
        return [e for item in items for e in self.monitor.ingest(item)]

    @property
    def observations(self):
        return [r for r in self.received if isinstance(r, Observation)]

    @property
    def heartbeats(self):
        return [r for r in self.received if isinstance(r, Heartbeat)]


def setup(tmp_path, scenario_id, seed=None):
    clock, pipe = FakeClock(), Pipeline()
    ctl = SimController(pipe.ingest, clock=clock, scenarios=SCENARIOS, live_dir=tmp_path)
    asyncio.run(ctl.load(scenario_id, seed))
    return ctl, clock, pipe


async def run_for(ctl, clock, seconds):
    for _ in range(round(seconds / DT)):
        clock.t += timedelta(seconds=DT)
        await ctl.tick(DT)


def drive(ctl, clock, *steps):
    """Each step is "play", "pause", ("speed", x), or a number of wall seconds to run."""

    async def go():
        for step in steps:
            if step == "play":
                await ctl.play(autorun=False)
            elif step == "pause":
                await ctl.pause()
            elif isinstance(step, tuple):
                await ctl.set_speed(step[1])
            else:
                await run_for(ctl, clock, step)

    asyncio.run(go())


def frame_t(obs, fps=12.0):
    """Scenario time of an observation, from its frame number in the obs_id."""
    parts = obs.obs_id.split("-")
    frame = parts[-2] if obs.person_id else parts[-1]
    return int(frame) / fps


def test_plays_in_real_time_with_scenario_spacing(tmp_path):
    ctl, clock, pipe = setup(tmp_path, "possible-fall-01")
    drive(ctl, clock, "play", 2.0)
    assert ctl.status().t == pytest.approx(2.0)
    obs = pipe.observations
    assert 20 <= len(obs) <= 25  # ~12 fps, minus the odd dropped frame
    stamps = [r.timestamp for r in pipe.received]
    assert stamps == sorted(stamps)
    first = obs[0]
    for o in obs:  # wall-clock stamps keep the scenario's spacing exactly
        assert (o.timestamp - first.timestamp).total_seconds() == pytest.approx(frame_t(o) - frame_t(first), abs=1e-5)


def test_speed_changes_pace_not_spacing(tmp_path):
    ctl, clock, pipe = setup(tmp_path, "possible-fall-01")
    drive(ctl, clock, ("speed", 2.0), "play", 1.0)
    assert ctl.status().t == pytest.approx(2.0)
    a, b = pipe.observations[0], pipe.observations[-1]
    assert (b.timestamp - a.timestamp).total_seconds() == pytest.approx(frame_t(b) - frame_t(a), abs=1e-5)
    with pytest.raises(ValueError):
        asyncio.run(ctl.set_speed(50))


def test_pause_sends_heartbeats_only_and_resumes_in_place(tmp_path):
    ctl, clock, pipe = setup(tmp_path, "possible-fall-01")
    drive(ctl, clock, "play", 2.0, "pause")
    n_obs, n_beats, t = len(pipe.observations), len(pipe.heartbeats), ctl.status().t
    drive(ctl, clock, 3.0)
    assert len(pipe.observations) == n_obs
    assert 2 <= len(pipe.heartbeats) - n_beats <= 4
    assert ctl.status().state == "paused" and ctl.status().t == t
    drive(ctl, clock, "play", 1.0)
    assert ctl.status().t == pytest.approx(t + 1.0)
    stamps = [r.timestamp for r in pipe.received]
    assert stamps == sorted(stamps)


def test_stamps_never_go_backwards_after_fast_play_and_resume(tmp_path):
    ctl, clock, pipe = setup(tmp_path, "possible-fall-01")
    drive(ctl, clock, ("speed", 4.0), "play", 2.0, "pause", 0.2, "play", 1.0)
    stamps = [r.timestamp for r in pipe.received]
    assert stamps == sorted(stamps)


def test_scripted_outage_is_silent_even_when_paused(tmp_path):
    ctl, clock, pipe = setup(tmp_path, "camera-outage-01")
    outage = next(g for g in ctl._live.run.ground_truth if g.truth_label == "camera_outage")
    drive(ctl, clock, "play", outage.t_start + 1.0, "pause")
    before = len(pipe.received)
    drive(ctl, clock, 3.0)
    assert len(pipe.received) == before
    assert [g.truth_label for g in ctl.status().truth] == ["camera_outage"]


def test_full_run_is_scored_for_the_after_action_view(tmp_path):
    ctl, clock, pipe = setup(tmp_path, "possible-fall-01")
    drive(ctl, clock, "play", 40.0)
    status = ctl.status()
    assert status.state == "ended"
    report = ctl.report(status.run_id)
    assert report.counts.true_positives == 1 and report.counts.false_positives == 0
    assert [r.run_id for r in ctl.runs()] == [status.run_id]
    assert (tmp_path / f"{status.run_id}.report.json").exists()


def test_pausing_does_not_change_measured_delay(tmp_path):
    """Pausing mid-fall to explain something must not change what the monitor concludes."""
    straight, clock_a, _ = setup(tmp_path / "a", "possible-fall-01")
    drive(straight, clock_a, "play", 40.0)
    paused, clock_b, _ = setup(tmp_path / "b", "possible-fall-01")
    drive(paused, clock_b, "play", 13.0, "pause", 5.0, "play", 40.0)

    def delays(c):
        return c.report(c.status().run_id).detection_delays

    assert delays(paused) == pytest.approx(delays(straight), abs=0.06)


def test_intentional_lying_down_is_not_flagged(tmp_path):
    ctl, clock, _ = setup(tmp_path, "lying-intentional-01")
    drive(ctl, clock, "play", 45.0)
    assert ctl.report(ctl.status().run_id).counts.false_positives == 0


def test_day_room_flags_the_right_resident(tmp_path):
    ctl, clock, _ = setup(tmp_path, "day-room-01")
    drive(ctl, clock, "play", 12.0)
    assert {t.person_id for t in ctl.status().truth} == {"p1", "p2", "p3", "p4"}
    drive(ctl, clock, 30.0)
    report = ctl.report(ctl.status().run_id)
    assert report.counts.true_positives == 1 and report.alerts[0].person_id == "p3"


def test_reset_replays_the_same_run(tmp_path):
    ctl, clock, pipe = setup(tmp_path, "walk-and-sit-01")
    drive(ctl, clock, "play", 3.0)
    first = [o.obs_id for o in pipe.observations]
    asyncio.run(ctl.reset())
    assert ctl.status().state == "loaded" and ctl.status().t == 0
    pipe.received.clear()
    drive(ctl, clock, "play", 3.0)
    assert [o.obs_id for o in pipe.observations] == first


def test_ground_truth_is_never_sent_to_ingest(tmp_path):
    ctl, clock, pipe = setup(tmp_path, "day-room-01")
    drive(ctl, clock, "play", 40.0)
    assert not any(isinstance(r, GroundTruth) for r in pipe.received)
    assert all(r.source_type == "simulation" for r in pipe.received)


def test_background_loop_runs_to_the_end(tmp_path):
    clock, pipe = FakeClock(), Pipeline()
    ctl = SimController(pipe.ingest, clock=clock, scenarios=SCENARIOS, live_dir=tmp_path)

    async def go():
        await ctl.load("possible-fall-01")
        await ctl.play()
        await ctl._task

    asyncio.run(go())
    assert ctl.status().state == "ended"
    assert ctl.report(ctl.status().run_id).counts.true_positives == 1


def test_scenario_list_describes_rooms_and_residents(tmp_path):
    ctl, _, _ = setup(tmp_path, "possible-fall-01")
    infos = {i.scenario_id: i for i in ctl.scenarios()}
    assert infos["day-room-01"].residents == ["p1", "p2", "p3", "p4"]
    assert infos["possible-fall-01"].residents == [] and infos["possible-fall-01"].scene.floor_y == 0.92


def test_generated_scenarios_are_registered_and_capped():
    from hearth.sim.generate import GenSpec, generate

    controller = SimController(Pipeline().ingest, clock=FakeClock(), scenarios=SCENARIOS, live_dir=None)
    controller.MAX_GENERATED = 3
    ids = [controller.register(generate(GenSpec(), seed)).scenario_id for seed in range(5)]
    known = {s.scenario_id for s in controller.scenarios()}
    assert set(ids[-3:]) <= known and not set(ids[:2]) & known
    assert set(SCENARIOS) <= known  # the library is never dropped
    assert not set(ids) & set(SCENARIOS)  # nor is the caller's dict changed
