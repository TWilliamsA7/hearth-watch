import json

import pytest
import stub_monitor  # noqa: F401  (resolves "stub_monitor:create_monitor" in this process and in workers)
from hearth.eval.__main__ import main
from hearth.eval.report import StudyReport
from hearth.eval.study import build_report, plan_study, run_study, study_markdown, wilson
from hearth.sim.behavior import FALL_KINDS
from hearth.sim.generate import GenSpec
from hearth.sim.scenario import load_scenario

STUB = "stub_monitor:create_monitor"


def test_wilson_interval():
    r = wilson(8, 10)
    assert r.rate == 0.8 and r.low == pytest.approx(0.4902, abs=1e-3) and r.high == pytest.approx(0.9433, abs=1e-3)
    assert wilson(0, 0).rate is None
    assert wilson(0, 5).low == 0.0 and wilson(5, 5).high == 1.0


def test_random_plan_draws_people_and_rooms_deterministically():
    plan = plan_study(GenSpec(), count=40, people=(1, 4), rooms=["day-room", "lounge"], seed=100)
    assert plan == plan_study(GenSpec(), count=40, people=(1, 4), rooms=["day-room", "lounge"], seed=100)
    assert [seed for _, seed in plan] == list(range(100, 140))
    assert {s.people for s, _ in plan} == {1, 2, 3, 4} and {s.room for s, _ in plan} == {"day-room", "lounge"}


def test_sweep_fills_every_cell_equally():
    plan = plan_study(GenSpec(fall_rate=1.0), count=24, people=(1, 3), rooms=["bedroom"], sweep=True)
    cells = {(s.people, s.room, next(iter(s.fall_kinds))) for s, _ in plan}
    assert len(cells) == 3 * len(FALL_KINDS) and len(plan) == 3 * len(FALL_KINDS) * 2


@pytest.fixture(scope="module")
def study():
    plan = plan_study(GenSpec(fall_rate=1.0, outage_rate=0.3), count=12, people=(1, 3),
                      rooms=["day-room", "bedroom", "lounge"], seed=3)  # fmt: skip
    outcomes = run_study(plan, STUB)
    return outcomes, build_report(outcomes, STUB, worst=20)


def test_breakdowns_add_up_to_the_totals(study):
    outcomes, report = study
    assert report.runs == 12
    assert report.counts.expected_alerts == sum(o.report.counts.expected_alerts for o in outcomes) >= 12
    for factor in ("fault", "people", "room"):
        rows = [b for b in report.falls_by if b.factor == factor]
        assert sum(b.detected.n for b in rows) == report.counts.expected_alerts
        assert sum(b.detected.hits for b in rows) == report.counts.true_positives
    activities = [b for b in report.false_alerts_by if b.factor == "activity"]
    assert sum(b.false_alerts for b in activities) == report.counts.false_positives
    rooms = [b for b in report.false_alerts_by if b.factor == "room"]
    assert sum(b.hours for b in rooms) == pytest.approx(report.simulated_hours, abs=1e-3)
    assert "Simulated scenarios only" in study_markdown(report)


def test_worst_runs_are_saved_for_replay(study, tmp_path):
    outcomes, _ = study
    report = build_report(outcomes, STUB, worst=20, reproduce_dir=tmp_path)
    for w in report.worst:
        assert w.missed or w.false_alerts
        path = w.reproduce.split(" run ", 1)[1]
        assert load_scenario(path).scenario_id == w.scenario_id


def test_study_cli_with_workers(tmp_path):
    out = tmp_path / "study"
    assert main(["study", "--count", "4", "--people", "2", "--jobs", "2", "--monitor", STUB, "--out", str(out)]) == 0
    report = StudyReport.model_validate(json.loads((out / "study.json").read_text(encoding="utf-8")))
    assert report.runs == 4 and len(list((out / "runs").glob("*.report.json"))) == 4
    assert (out / "study.md").exists()
