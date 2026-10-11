"""Headless live runs: written like scripted runs, scored by the same evaluator."""

import json

from hearth.eval.batch import PerPersonMonitor, drive, load_monitor_factory
from hearth.eval.scoring import evaluate
from hearth.sim.__main__ import main
from hearth.sim.live.runner import LiveScenario, load_live_scenarios, run_live
from hearth.sim.live.spec import EngineSpec
from hearth.sim.runner import write_run


def test_a_live_run_writes_the_usual_files(tmp_path) -> None:
    run = run_live(LiveScenario(seed=2, duration=20, spec=EngineSpec(actors=3)))
    paths = write_run(run, tmp_path)  # type: ignore[arg-type]
    info = json.loads(paths["run"].read_text())
    assert info["run_id"] == "live-2-s2" and info["residents"] == ["p1", "p2", "p3"] and info["scene"]["objects"]
    world = json.loads(paths["world"].read_text())
    assert world["room"]["bounds"] and len(world["frames"]) >= 190
    truth = [json.loads(line) for line in paths["truth"].read_text().splitlines()]
    assert {t["person_id"] for t in truth} == {"p1", "p2", "p3"}


def test_the_shipped_live_scenarios_load_and_the_demo_fall_is_detected() -> None:
    library = load_live_scenarios()
    assert {"live-day-room-01", "live-fall-01"} <= set(library)
    run = run_live(library["live-fall-01"], frames=False)
    events = drive(run, PerPersonMonitor(load_monitor_factory()), tail=6)  # type: ignore[arg-type]
    report = evaluate(events, run.ground_truth, run.start)
    assert report.counts.expected_alerts == 1 and report.counts.true_positives == 1
    assert report.counts.false_positives == 0


def test_cli(tmp_path, capsys) -> None:
    assert main(["live", "--people", "2", "--seed", "1", "--duration", "10", "--out", str(tmp_path)]) == 0
    assert "live-1-s1" in capsys.readouterr().out
    assert (tmp_path / "live-1-s1.world.json").exists()
