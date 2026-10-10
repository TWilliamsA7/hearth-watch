from datetime import timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hearth_schema import PosturePrediction
from hearth.perception.fall_model import ModelSettings, canonical_label, prediction_from_detections, observation_from_result
from conftest import BASE, FLOOR, observation

SETTINGS = ModelSettings.load()


def prediction(label="fallen", confidence=.9, **changes):
    data = dict(model_id=SETTINGS.repo_id, model_revision=SETTINGS.revision, label=label,
                confidence=confidence, detection_count=1, bbox=dict(x1=.1,y1=.2,x2=.8,y2=.9))
    data.update(changes)
    return PosturePrediction(**data)


def frame(t, label="fallen", confidence=.9, **changes):
    return observation(t, features=None, posture_prediction=prediction(label,confidence), **changes)


def falls(service):
    return [e for e in service.events if e.event_type=="possible_fall"]


def test_real_model_label_names_are_mapped_not_numeric_class_ids():
    assert [canonical_label(s) for s in ("Standing","Sitting","Fallen")] == ["standing","sitting","fallen"]
    assert canonical_label("other") == "unknown"


def test_single_box_normalized_and_no_fake_pose():
    class Tensor:
        def __init__(self, values):self.values=values
        def cpu(self):return self
        def tolist(self):return self.values
    result=SimpleNamespace(orig_shape=(100,200),names={7:"Standing"},
                           boxes=SimpleNamespace(xyxy=Tensor([[20,10,100,90]]),cls=Tensor([7]),conf=Tensor([.95])))
    obs=observation_from_result(result,timestamp=BASE,source="model",source_type="edge",room_id="room-01",run_id="one",settings=SETTINGS)
    assert obs.person_detected and obs.posture_prediction.label=="standing"
    assert obs.posture_prediction.bbox.x1==.1 and obs.posture_prediction.bbox.y2==.9
    assert obs.features is None and obs.keypoints is None and obs.face_present is None


@pytest.mark.parametrize("detections,count,reason",[
    ([],0,"no_detection"),
    ([dict(label="Standing",confidence=.9,xyxy=[0,0,40,100]),dict(label="Fallen",confidence=.9,xyxy=[50,50,100,100])],2,"multiple_people"),
    ([dict(label="other",confidence=.9,xyxy=[0,0,50,100])],1,"unsupported_label"),
    ([dict(label="Standing",confidence=.1,xyxy=[0,0,50,100])],0,"no_detection")])
def test_ambiguous_or_missing_detections_are_unknown(detections,count,reason):
    p=prediction_from_detections(detections,100,100,SETTINGS)
    assert p.label=="unknown" and p.detection_count==count and p.uncertainty_reason==reason


def test_sustained_fallen_creates_alert_then_sustained_standing_recovers(service):
    service.ingest([frame(t/2) for t in range(6)])
    assert not falls(service)
    service.ingest([frame(3),frame(3.5)])
    assert len(falls(service))==1
    event=falls(service)[0]
    assert event.metadata["model_id"]==SETTINGS.repo_id
    assert event.metadata["reason"]=="sustained_model_fallen" and event.confidence==.9
    service.ingest([frame(4,"standing"),frame(4.5,"standing")])
    alert=next(iter(service.alerts.values()))
    assert alert.recovered_at is None
    service.ingest([frame(5,"standing")])
    assert alert.recovered_at==BASE+timedelta(seconds=5) and alert.status=="open"


@pytest.mark.parametrize("label,confidence",[("standing",.9),("sitting",.9),("fallen",.4)])
def test_non_fall_and_low_confidence_predictions_do_not_alert(service,label,confidence):
    service.ingest([frame(t/2,label,confidence) for t in range(10)])
    assert not falls(service)


def test_prediction_presence_blocks_conflicting_pose_fallback(service):
    service.ingest([observation(t/2,FLOOR,posture_prediction=prediction("standing")) for t in range(10)])
    assert not falls(service)


def test_changing_evidence_source_resets_pending_hold(service):
    service.ingest([observation(t/2,FLOOR) for t in range(4)])
    service.ingest([frame(t/2) for t in range(4,10)])
    assert not falls(service)
    service.ingest([frame(5)])
    assert len(falls(service))==1


def test_unknown_predictions_and_sitting_never_recover(service):
    service.ingest([frame(t/2) for t in range(7)])
    unknown=PosturePrediction(model_id=SETTINGS.repo_id,model_revision=SETTINGS.revision,
                             label="unknown",detection_count=2,uncertainty_reason="multiple_people")
    service.ingest([observation(3.5,None,posture_prediction=unknown),frame(4,"sitting"),frame(4.5,"standing",.3)])
    assert next(iter(service.alerts.values())).recovered_at is None


def test_prediction_gaps_and_unknown_frames_reset_hold(service):
    service.ingest([frame(0),frame(.5),frame(10),frame(10.5),frame(11)])
    assert not falls(service)
    service.ingest([frame(11.5,"fallen",.2),frame(12),frame(12.5),frame(13)])
    assert not falls(service)


def test_source_status_exposes_prediction_and_staleness_clears_it(service,clock):
    service.ingest([frame(0,"standing")])
    source=service.snapshot().sources[0]
    assert source.posture=="standing" and source.posture_model==SETTINGS.repo_id
    clock.seconds=4
    source=service.snapshot().sources[0]
    assert source.posture=="unknown" and source.posture_confidence is None and source.observation_stale


def test_wrong_model_cannot_decide_fall(service):
    service.ingest([observation(t/2,None,posture_prediction=prediction(model_id="different-model")) for t in range(10)])
    assert not falls(service)


def test_model_schema_rejects_inconsistent_prediction():
    with pytest.raises(ValidationError):prediction(detection_count=2)
    with pytest.raises(ValidationError):prediction(uncertainty_reason="multiple_people")
    with pytest.raises(ValidationError):prediction(bbox=dict(x1=.8,y1=.2,x2=.1,y2=.9))
