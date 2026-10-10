import json

import pytest
import stub_monitor
from hearth.eval.__main__ import main
from hearth.eval.batch import drive, evaluate_run, load_monitor_factory, run_batch
from hearth.eval.scoring import summarize
from hearth.sim.runner import run_headless
from hearth.sim.scenario import load_all

SCENARIOS = load_all()
STUB = "stub_monitor:create_monitor"


def test_drive_advances_the_clock_through_silence():
    """During a camera outage nothing arrives, so only advance_to can raise source_offline."""
    run = run_headless(SCENARIOS["camera-outage-01"])
    events = drive(run, stub_monitor.create_monitor())
    offline = [e for e in events if e.event_type == "source_offline"]
    # One during the outage, one after the scenario ends and the simulated source stops for good.
    assert len(offline) == 2
    assert 10.0 <= (offline[0].timestamp - run.start).total_seconds() <= 11.0  # outage starts at 7 s, 3 s threshold


def test_batch_scores_through_the_seam():
    reports = run_batch(SCENARIOS.values(), seeds=2, factory=stub_monitor.create_monitor)
    by_id = {r.run_id: r for r in reports}
    assert len(reports) == 2 * len(SCENARIOS)
    assert by_id["possible-fall-01-s42"].counts.true_positives == 1
    assert by_id["lying-intentional-01-s23"].counts.false_positives == 0
    outage = next(s for s in by_id["camera-outage-01-s5"].segments if s.truth_label == "camera_outage")
    assert outage.detected
    assert summarize(reports).runs == len(reports)


def test_each_run_gets_a_fresh_monitor():
    calls = []

    def factory():
        calls.append(1)
        return stub_monitor.create_monitor()

    run_batch([SCENARIOS["possible-fall-01"]], seeds=3, factory=factory)
    assert len(calls) == 3


def test_batch_is_deterministic():
    a, _ = evaluate_run(SCENARIOS["possible-fall-01"], 42, stub_monitor.create_monitor)
    b, _ = evaluate_run(SCENARIOS["possible-fall-01"], 42, stub_monitor.create_monitor)
    assert a == b


def test_missing_monitor_fails_with_a_clear_message():
    with pytest.raises(RuntimeError, match="ingest"):
        load_monitor_factory("hearth.monitor_does_not_exist:create_monitor")


def test_cli_batch_writes_reports(tmp_path):
    assert main(["batch", "--seeds", "1", "--monitor", STUB, "--out", str(tmp_path)]) == 0
    assert len(list((tmp_path / "runs").glob("*.report.json"))) == len(SCENARIOS)
    assert "Simulated scenarios only" in (tmp_path / "summary.md").read_text()
    assert json.loads((tmp_path / "summary.json").read_text())["runs"] == len(SCENARIOS)


def test_cli_scores_a_captured_websocket_log(tmp_path, capsys):
    """An event log captured from the live WebSocket (ServerMessages) scores the same as a batch run."""
    from hearth.sim.runner import write_run

    run = run_headless(SCENARIOS["possible-fall-01"])
    paths = write_run(run, tmp_path)
    events = drive(run, stub_monitor.create_monitor())
    log = tmp_path / "ws.jsonl"
    lines = [json.dumps({"type": "event", "data": e.model_dump(mode="json")}) for e in events]
    lines.insert(0, json.dumps({"type": "source_status", "data": {}}))
    log.write_text("\n".join(lines))

    assert main(["score", "--run", str(paths["run"]), "--events", str(log)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["counts"]["true_positives"] == 1


# --- several residents ---


def test_each_person_gets_their_own_monitor_and_alerts_name_them():
    report, events = evaluate_run(SCENARIOS["day-room-01"], None, stub_monitor.create_monitor)
    falls = [e for e in events if e.event_type == "possible_fall"]
    assert [e.person_id for e in falls] == ["p3"]
    assert report.counts.true_positives == 1 and report.counts.false_positives == 0


def test_one_monitor_for_a_whole_crowd_cannot_attribute_alerts():
    """Why the adapter exists: a person-unaware monitor mixes everyone's keypoints together."""
    report, _ = evaluate_run(SCENARIOS["day-room-01"], None, stub_monitor.create_monitor, per_person=False)
    assert report.counts.true_positives == 0


def test_camera_events_are_reported_once_not_per_person():
    from datetime import timedelta

    from hearth.eval.batch import PerPersonMonitor
    from hearth.sim.generator import SIM_EPOCH
    from hearth_schema.models import Event, Observation

    class Loud:
        def ingest(self, item):
            return []

        def advance_to(self, now):
            return [Event(event_id=f"x-{id(self)}", timestamp=now, source="sim", source_type="simulation",
                          room_id="room-01", event_type="source_offline")]

    m = PerPersonMonitor(Loud)
    for pid in ("p1", "p2", "p3"):
        m.ingest(Observation(obs_id=pid, timestamp=SIM_EPOCH, source="sim", source_type="simulation",
                             room_id="room-01", person_id=pid, person_detected=True))
    assert len(m.advance_to(SIM_EPOCH + timedelta(seconds=5))) == 1
