"""Live admission uses camera evidence; render ticks cannot confirm or prolong a track."""
from types import SimpleNamespace

import pytest
from hearth_schema.models import Observation

from hearth.monitor.config import MonitorConfig
from hearth.monitor.service import MonitorService
from hearth.sim.live.engine import World
from hearth.sim.live.link import LinkRequest, LiveLink
from hearth.sim.live.spec import EngineSpec
from test_live_link import SOURCE, SPEC, camera_frames, people_frames


@pytest.fixture
def linked():
    now = [1000.0]
    monitor = MonitorService(MonitorConfig.load(), clock=lambda: now[0])
    controller = SimpleNamespace(world=World(EngineSpec.model_validate(SPEC), 1),
                                 tick_hooks=[], keep_running=lambda: None)
    link = LiveLink(monitor, controller)
    link.configure(LinkRequest(enabled=True, source=SOURCE))

    def send(people=("p1",), start=0, seconds=2, **updates):
        frames = people_frames(controller.world,
                               [(pid, "standing", 0.4, -3.0) for pid in people], start, seconds)
        monitor.ingest([Observation.model_validate({**frame, **updates}) for frame in frames])
        link.update()

    return link, monitor, controller, now, send


def test_brief_detection_and_repeated_ticks_never_create_a_person(linked):
    link, _, controller, now, send = linked
    send(seconds=0.5)
    for _ in range(50):
        now[0] += 0.05
        link.update()
    assert not link.tracks and not link.candidates and not controller.world.externals


def test_batched_evidence_confirms_three_with_stable_priority(linked):
    link, _, controller, now, send = linked
    send(people=("p4", "p3", "p2", "p1"))
    assert list(link.tracks) == ["p1", "p2", "p3"]
    assert len(controller.world.externals) == 3
    assert link.status().max_people == 3
    # A waiting fourth person stays visible while the original three disappear.
    for i in range(1, 42):
        now[0] += 0.1
        send(people=("p4",), start=1.9 + i / 10, seconds=0.1)
        assert len(link.tracks) <= 3
    assert list(link.tracks) == ["p4"]


def test_duplicate_frames_do_not_add_evidence(linked):
    link, _, _, now, send = linked
    send(seconds=0.1)
    for _ in range(10):
        now[0] += 0.1
        send(seconds=0.1)
    assert not link.tracks


def test_old_candidate_in_batch_cannot_be_admitted_after_source_moves_on(linked):
    link, monitor, controller, _, _ = linked
    frames = people_frames(controller.world, [("p1", "standing", 0.4, -3.0)], 0, 2)
    frames += people_frames(controller.world, [("p2", "standing", 0.4, -3.0)], 4, 2)
    monitor.ingest([Observation.model_validate(frame) for frame in frames])
    link.update()
    assert list(link.tracks) == ["p2"]


@pytest.mark.parametrize("interruption", ["gap", "absent", "no_keypoints", "nobody", "run"])
def test_candidate_confirmation_resets_on_interruption(linked, interruption):
    link, _, _, _, send = linked
    send(seconds=1)
    if interruption == "gap":
        send(start=1.5, seconds=0.7)
    elif interruption == "run":
        send(start=1, seconds=0.7, run_id="r2")
    else:
        updates = {"keypoints": None} if interruption == "no_keypoints" else {"person_detected": False}
        if interruption == "nobody":
            updates["person_id"] = None
        send(start=1, seconds=0.1, **updates)
        send(start=1.1, seconds=0.7)
    assert not link.tracks


def test_confirmation_requires_duration_and_five_observations(linked):
    link, _, _, _, send = linked
    for t in (0, 0.5, 1.0, 1.5):
        send(start=t, seconds=0.1)
    assert not link.tracks
    send(start=1.6, seconds=0.1)
    assert list(link.tracks) == ["p1"]


def test_confirmed_track_survives_dropout_but_expires_from_actual_receipt(linked):
    link, _, controller, now, send = linked
    send()
    original = link.tracks["p1"]
    now[0] += 0.8
    link.update()
    assert link.tracks["p1"] is original
    send(start=2.7, seconds=0.1)
    assert link.tracks["p1"] is original
    for _ in range(41):
        now[0] += 0.1
        link.update()
    assert not link.tracks and not controller.world.externals
    send(start=7, seconds=0.1)
    assert not link.tracks  # an expired id must confirm again


def test_missing_person_observations_cannot_extend_confirmed_lifetime(linked):
    link, _, controller, now, send = linked
    send()
    for i in range(1, 42):
        now[0] += 0.1
        send(start=2 + i / 10, seconds=0.1, person_detected=False)
    assert not link.tracks and not controller.world.externals


def test_link_changes_clear_evidence_and_preserve_limit(linked):
    link, _, controller, _, send = linked
    send(people=("p1", "p2", "p3"))
    link.configure(LinkRequest(enabled=True, source=SOURCE, max_people=1))
    assert list(link.tracks) == ["p1"] and len(controller.world.externals) == 1
    link.configure(LinkRequest(enabled=True, source=SOURCE, height=1.8))
    assert link.max_people == 1
    link.configure(LinkRequest(enabled=False, source=SOURCE))
    link.configure(LinkRequest(enabled=True, source=SOURCE))
    link.update()
    assert not link.tracks and not link.candidates and not controller.world.externals
    send(start=2, seconds=1)
    link.configure(LinkRequest(enabled=True, source="jetson-01"))
    assert not link.tracks and not link.candidates and link.max_people == 1


def test_room_replacement_and_run_change_require_new_confirmation(linked):
    link, _, controller, _, send = linked
    send()
    controller.world = World(EngineSpec.model_validate(SPEC), 2)
    assert link.status().people == []
    send(start=2, seconds=0.5)
    assert not link.tracks
    send(start=2.5)
    assert link.tracks
    send(start=0, seconds=0.5, run_id="new-run")
    assert not link.tracks and not controller.world.externals


def test_linking_does_not_admit_cached_observations(linked):
    link, _, _, _, send = linked
    link.configure(LinkRequest(enabled=False, source=SOURCE))
    send()
    link.configure(LinkRequest(enabled=True, source=SOURCE))
    link.update()
    assert not link.tracks and not link.candidates


def test_single_person_feed_confirms_without_a_track_id(linked):
    link, monitor, controller, _, _ = linked
    frames = camera_frames(controller.world, "standing", 0.4, -3.0, 0, 2)
    monitor.ingest([Observation.model_validate(frame) for frame in frames])
    link.update()
    assert list(link.tracks) == [None]
    assert list(controller.world.externals) == [f"live-{SOURCE}"]


@pytest.mark.parametrize("limit", [0, 11, 1.5, True, "3"])
def test_limit_validation(limit):
    with pytest.raises(ValueError):
        LinkRequest(enabled=True, max_people=limit)
