"""A live person in the simulated room: residents see their body; nothing else is simulated for them."""

import math

from hearth.sim.live.activities import Walk
from hearth.sim.live.engine import World
from hearth.sim.live.lift import lift
from hearth.sim.live.spec import CameraSpec, EngineSpec, IncidentSpec, RoomSpec
from hearth.sim.nav import distance
from hearth.sim.pose import KEY_POSES
from hearth.sim.world import body

SPEC = EngineSpec(actors=1, noise=False, room=RoomSpec(width=6.5, depth=5.5, furniture_density=0.0),
                  camera=CameraSpec(preset="jetson"))


def live_body(world, x, z, pose="standing", posture="standing"):
    """What the Jetson would report for someone at (x, z), lifted back into the room."""
    truth = body(KEY_POSES[pose], 0.0, x, z, 0.0, 1.7)
    pts = [(n, *q[:2]) for n, q in ((n, world.camera.project(p)) for n, p in truth.items())
           if q is not None and 0 <= q[0] <= 1 and 0 <= q[1] <= 1]
    return lift(pts, world.camera, world.room, posture=posture)


def test_live_person_is_in_the_frame_but_not_in_truth_or_the_sim_stream():
    w = World(SPEC, 1)
    w.set_external("jetson-01", live_body(w, 0.0, -2.5))
    frame = w.frame()
    live = [p for p in frame.people if p.kind == "live"]
    assert [p.person_id for p in live] == ["live-jetson-01"] and live[0].activity.startswith("live")
    records = w.advance(3.0)
    assert not any(getattr(r, "person_id", None) == "live-jetson-01" for r in records)
    assert all(gt.person_id != "live-jetson-01" for gt in w.ground_truth())


def test_gone_means_absent():
    w = World(SPEC, 1)
    w.set_external("jetson-01", live_body(w, 0.0, -2.5))
    w.set_external("jetson-01", None)
    assert all(p.kind != "live" for p in w.frame().people)


def test_residents_walk_around_a_live_person():
    w = World(SPEC, 1)
    a = w.actors[0]
    a.pos, a.posture = (0.0, -1.2), "standing"
    w.set_external("jetson-01", live_body(w, 0.0, -2.7))
    w.replace_plan(a, [Walk((0.0, -4.4))], "test")
    closest = math.inf
    for _ in range(int(12 / w.dt)):
        w.step()
        closest = min(closest, distance(a.pos, w.externals["live-jetson-01"].pos))
    assert closest > 0.35


def test_a_live_person_blocks_the_simulated_camera():
    def seen_joints(with_live):
        w = World(SPEC, 1)
        a = w.actors[0]
        a.pos, a.heading, a.posture, a.plan = (0.0, -4.0), 0.0, "standing", type(a.plan)()
        if with_live:
            w.set_external("jetson-01", live_body(w, 0.0, -2.0))
        obs = [r for r in w.advance(1.0) if getattr(r, "person_id", None) == a.person_id]
        return max((len(o.keypoints or []) for o in obs), default=0)

    assert seen_joints(with_live=True) < seen_joints(with_live=False)


def test_a_resident_comes_to_help_a_live_person_on_the_floor():
    w = World(SPEC.model_copy(update={"actors": 2}), 1)
    for a in w.actors:
        a.posture = "standing"
    e = w.set_external("jetson-01", live_body(w, 0.5, -3.0, "side_lying", "floor"), posture="floor", label="fall")
    helper = w.incidents.respond(e)
    assert helper is not None and helper.intent == "help"
    for _ in range(int(20 / w.dt)):
        w.step()
    assert distance(helper.pos, e.pos) < 1.45 and helper.intent == "help"  # beside them, past the clearance
    w.set_external("jetson-01", live_body(w, 0.5, -3.0), posture="standing")  # they got up
    for _ in range(int(6 / w.dt)):
        w.step()
    assert "live-jetson-01" not in w.incidents.responders


def test_simulated_falls_get_a_helper_too():
    w = World(EngineSpec(actors=3, noise=False, incidents=[IncidentSpec(kind="fall", t=2.0, recover=False)]), 4)
    w.advance(15.0)
    assert w.incidents.responders
