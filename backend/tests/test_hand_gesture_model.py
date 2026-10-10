import pytest
from fastapi.testclient import TestClient

from hearth.api.app import create_app
from conftest import observation

MODEL = "SuperSura/hand-gesture-yolo26n"


def frame(t, label="fist", **changes):
    prediction = dict(model_id=MODEL, model_revision="test", label=label, confidence=.9,
                      bbox=dict(x1=.1,y1=.1,x2=.3,y2=.4), detection_count=1)
    prediction.update(changes)
    return observation(t, features=None, person_detected=False, hand_gesture_prediction=prediction)


def test_hold_routes_help_to_subscribed_nurse_and_latches(service):
    with TestClient(create_app(service=service)) as client:
        nurse = client.post('/nurse-sessions', json=dict(display_name="Sam",room_ids=["room-01"],
                    alert_kinds=["help_requested"],on_duty=True)).json()
        response = client.post('/ingest/observations', json={'observations':[
            frame(i/2).model_dump(mode='json') for i in range(4)]})
        assert response.json()['accepted'] == 4
        assert not service.alerts
        queue = service.subscribe(nurse_id=nurse['nurse_id'])
        response = client.post('/ingest/observations', json={'observations':[
            frame(t).model_dump(mode='json') for t in (2,2.5,3)]})
        assert response.json()['accepted'] == 3
        alert = next(iter(service.alerts.values()))
        assert alert.kind == "help_requested" and alert.metadata['method'] == "hand_gesture"
        assert alert.metadata['gesture'] == "fist"
        assert service.nurse_snapshot(nurse['nurse_id']).alerts == [alert]
        messages = [queue.get_nowait() for _ in range(queue.qsize())]
        assert any(m['type']=='alert' and m['data']['kind']=='help_requested' for m in messages)
        service.ingest([frame(3.5,'unknown',confidence=None,bbox=None,detection_count=0,uncertainty_reason='no_detection'),frame(4)])
        assert len(service.alerts) == 1


@pytest.mark.parametrize('changes', [dict(label='palm'), dict(confidence=.2), dict(model_id='wrong'),
    dict(label='unknown',confidence=None,bbox=None,detection_count=2,uncertainty_reason='multiple_hands')])
def test_other_gestures_weak_or_ambiguous_detections_do_not_alert(service, changes):
    service.ingest([frame(i/2, **changes) for i in range(8)])
    assert not service.alerts


def test_hand_result_uses_checkpoint_names_and_normalizes_boxes():
    from types import SimpleNamespace
    from hearth.perception.hand_model import settings, prediction_from_result
    class Tensor:
        def __init__(self, values): self.values = values
        def cpu(self): return self
        def tolist(self): return self.values
    result = SimpleNamespace(orig_shape=(100,200), names={7:'palm'}, boxes=SimpleNamespace(
        xyxy=Tensor([[20,10,60,40]]),cls=Tensor([7]),conf=Tensor([.9])))
    prediction = prediction_from_result(result, settings())
    assert prediction.label == 'palm' and prediction.bbox.x1 == .1
    assert prediction.bbox.x2 == .3 and prediction.detection_count == 1
    result.boxes = SimpleNamespace(xyxy=Tensor([[20,10,60,40],[100,10,140,40]]),
                                  cls=Tensor([7,7]),conf=Tensor([.9,.9]))
    assert prediction_from_result(result, settings()).uncertainty_reason == 'multiple_hands'


def test_gap_and_large_hand_movement_reset_pending_hold(service):
    service.ingest([frame(0),frame(.5),frame(2),frame(2.5),
        frame(3,bbox=dict(x1=.7,y1=.6,x2=.9,y2=.9)),frame(3.5)])
    assert not service.alerts


def test_observed_release_then_cooldown_allows_next_request(service):
    service.ingest([frame(i/2) for i in range(5)])
    service.ingest([frame(t,'palm') for t in (2.5,3,3.5)])
    service.ingest([frame(i/2) for i in range(8,25)])
    assert len(service.alerts) == 2


def test_missing_hand_prediction_overrides_pose_gesture_when_present(service):
    from test_help_and_nurses import points
    frames = [frame(i/2,'unknown',confidence=None,bbox=None,detection_count=0,uncertainty_reason='no_detection') for i in range(8)]
    for obs in frames:
        obs.person_detected = True
        from hearth_schema import Keypoint
        obs.keypoints = [Keypoint(**p) for p in points()]
    service.ingest(frames)
    assert not service.alerts
