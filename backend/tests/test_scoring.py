from datetime import timedelta

import pytest
from hearth.eval.scoring import EvalConfig, evaluate, summarize
from hearth.sim.generator import SIM_EPOCH
from hearth.sim.scenario import Variation, load_all
from hearth_schema.models import Event



def nominal_truth(scenario_id):
    return load_all()[scenario_id].model_copy(update={"variation": Variation.none()}).ground_truth()


FALL = nominal_truth("possible-fall-01")  # walking 0-6, standing 6-12, fall 12-30
OUTAGE = nominal_truth("camera-outage-01")  # camera_outage 7-15


def ev(t, event_type="possible_fall", source="sim", scenario_id="possible-fall-01", n=[0]):
    n[0] += 1
    return Event(event_id=f"evt-{n[0]}", timestamp=SIM_EPOCH + timedelta(seconds=t), source=source,
                 source_type="simulation", room_id="room-01", scenario_id=scenario_id, event_type=event_type)


def test_detected_fall_reports_delay():
    r = evaluate([ev(14.5)], FALL, SIM_EPOCH)
    assert r.counts.model_dump() == dict(expected_alerts=1, true_positives=1, false_positives=0, missed=0, duplicates=0)
    assert r.detection_delays == [2.5]
    fall = next(s for s in r.segments if s.truth_label == "fall")
    assert fall.detected and fall.delay == 2.5


def test_no_alert_is_a_miss():
    r = evaluate([], FALL, SIM_EPOCH)
    assert r.counts.missed == 1 and r.counts.true_positives == 0
    assert next(s for s in r.segments if s.truth_label == "fall").detected is False


def test_alert_outside_any_fall_is_a_false_positive_with_label():
    r = evaluate([ev(8.0), ev(13.0)], FALL, SIM_EPOCH)
    assert (r.counts.false_positives, r.counts.true_positives) == (1, 1)
    fp = next(a for a in r.alerts if a.outcome == "false_positive")
    assert fp.truth_label == "standing"
    assert next(s for s in r.segments if s.truth_label == "standing").false_alerts == 1


def test_repeat_alerts_are_duplicates_not_false_positives():
    r = evaluate([ev(13.0), ev(20.0)], FALL, SIM_EPOCH)
    assert (r.counts.true_positives, r.counts.duplicates, r.counts.false_positives) == (1, 1, 0)


def test_tolerance_after_segment_end():
    truth = nominal_truth("possible-fall-01")
    truth[-1] = truth[-1].model_copy(update={"t_end": 20.0})
    assert evaluate([ev(22.0)], truth, SIM_EPOCH, config=EvalConfig(alert_tolerance=3.0)).counts.true_positives == 1
    assert evaluate([ev(24.0)], truth, SIM_EPOCH, config=EvalConfig(alert_tolerance=3.0)).counts.false_positives == 1


def test_other_sources_and_scenarios_are_ignored():
    r = evaluate([ev(13.0, source="jetson-01"), ev(13.0, scenario_id="other")], FALL, SIM_EPOCH)
    assert r.counts.true_positives == 0 and r.timeline == []


def test_camera_outage_expects_source_offline():
    events = [ev(10.2, "source_offline", scenario_id=None), ev(15.4, "source_online", scenario_id=None)]
    outage = next(s for s in evaluate(events, OUTAGE, SIM_EPOCH).segments if s.truth_label == "camera_outage")
    assert outage.expected_event == "source_offline" and outage.detected and outage.delay == pytest.approx(3.2)
    missing = next(s for s in evaluate([], OUTAGE, SIM_EPOCH).segments if s.truth_label == "camera_outage")
    assert missing.detected is False


def test_timeline_is_on_the_scenario_clock():
    r = evaluate([ev(13.0), ev(2.0, "person_visible")], FALL, SIM_EPOCH)
    assert [(e.t, e.event_type) for e in r.timeline] == [(2.0, "person_visible"), (13.0, "possible_fall")]


def test_summary_totals_and_labels():
    reports = [evaluate([ev(13.0)], FALL, SIM_EPOCH), evaluate([ev(8.0)], FALL, SIM_EPOCH)]
    s = summarize(reports)
    assert (s.runs, s.counts.true_positives, s.counts.false_positives, s.counts.missed) == (2, 1, 1, 1)
    assert s.delay_median == 1.0 and s.false_positives_by_label == {"standing": 1}


# --- several residents ---

DAY = load_all()["day-room-01"].model_copy(update={"variation": Variation.none()}).ground_truth()
P3_FALL = next(g for g in DAY if g.expects_alert)  # p3 falls at 10 s


def pev(t, person_id, event_type="possible_fall"):
    return ev(t, event_type, scenario_id="day-room-01").model_copy(update={"person_id": person_id})


def test_alert_for_the_right_person_counts():
    r = evaluate([pev(P3_FALL.t_start + 2, "p3")], DAY, SIM_EPOCH)
    assert (r.counts.true_positives, r.counts.false_positives, r.counts.missed) == (1, 0, 0)
    assert r.alerts[0].person_id == "p3" and r.people == ["p1", "p2", "p3", "p4"]


def test_alert_for_the_wrong_person_is_a_false_alert_and_a_miss():
    r = evaluate([pev(P3_FALL.t_start + 2, "p4")], DAY, SIM_EPOCH)
    assert (r.counts.true_positives, r.counts.false_positives, r.counts.missed) == (0, 1, 1)
    assert r.alerts[0].truth_label == "sitting"  # p4 was on the couch at the time


def test_alert_without_a_person_cannot_be_attributed_in_a_crowd():
    r = evaluate([pev(P3_FALL.t_start + 2, None)], DAY, SIM_EPOCH)
    assert (r.counts.false_positives, r.counts.missed) == (1, 1)


def test_person_not_visible_must_name_the_absent_person():
    absent = next(g for g in DAY if g.person_id == "p2" and g.truth_label == "absent")
    right = evaluate([pev(absent.t_start + 1, "p2", "person_not_visible")], DAY, SIM_EPOCH)
    wrong = evaluate([pev(absent.t_start + 1, "p1", "person_not_visible")], DAY, SIM_EPOCH)
    seg = lambda r: next(s for s in r.segments if s.segment_id == absent.segment_id)  # noqa: E731
    assert seg(right).detected and not seg(wrong).detected
