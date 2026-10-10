"""Live simulation engine: a procedurally generated room in metres, ticking in real time.

Unlike the scripted pipeline (hearth.sim.scenario / behavior / generator), nothing here
is planned up front. Every actor runs its own state machine each tick: it picks an
activity, plans a route on the room's NavGrid, steers around other people, sits, rests,
talks, leaves and comes back. Incidents can be triggered at any moment. The camera
watches the same world and produces contract Observations, and ground truth is written
as it happens; both match what the scripted pipeline produces, so the monitor, scoring
and dashboard views treat a live run like any other.

    room.py        procedural room: walls, doors, window, furniture with slots
    actor.py       who each person is (profile, appearance) and their runtime state
    activities.py  the state machine's primitive activities and the Brain that picks plans
    social.py      conversations between two or more people
    animate.py     body poses from posture, gait and idle layers
    incidents.py   manual, scheduled and procedural incidents
    camera.py      pinhole camera: keypoints, visibility, occlusion, noise
    truth.py       ground-truth segments as labels change
    engine.py      World: the fixed-step loop tying them together
    spec.py        EngineSpec: everything that configures a run
"""
