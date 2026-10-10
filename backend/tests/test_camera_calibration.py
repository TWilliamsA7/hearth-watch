import pytest
from conftest import observation
from hearth.monitor.config import MonitorConfig
from hearth.monitor.detector import FallDetector, track_key


def test_mid_frame_floor_alert_and_standing_recovery_use_independent_bounds():
    config = MonitorConfig.load()
    detector = FallDetector(config)
    lying = dict(torso_angle_deg=72, head_y=.46, hip_y=.58, bbox_aspect=2.4)
    standing = dict(torso_angle_deg=12, head_y=.25, hip_y=.60, bbox_aspect=.4)
    events = []
    for i in range(32):
        events += detector.process(observation(i/10, lying))
    assert len([e for e in events if e.event_type == 'possible_fall']) == 1
    events = detector.process(observation(3.2, standing))
    assert [e.event_type for e in events] == ['recovered']
    assert detector.states[track_key(observation(0))].posture == 'upright'


@pytest.mark.parametrize('features', [
    dict(torso_angle_deg=12,head_y=.25,hip_y=.60,bbox_aspect=.4),
    dict(torso_angle_deg=25,head_y=.40,hip_y=.62,bbox_aspect=.7),
    dict(torso_angle_deg=72,head_y=.46,hip_y=.58,bbox_aspect=2.4)])
def test_floor_calibration_does_not_change_independent_upright_rule(features):
    c = MonitorConfig(posture_mode='pose_fallback',floor_head_y_min=.40,floor_hip_y_min=.45)
    det = FallDetector(c)
    events = []
    for i in range(32):
        events += det.process(observation(i/10,features))
    assert bool([e for e in events if e.event_type == 'possible_fall']) == (features['torso_angle_deg'] > 30)


def test_floor_calibration_does_not_override_confident_model_sitting():
    from test_fall_model import prediction
    det = FallDetector(MonitorConfig.load())
    events = []
    for i in range(32):
        obs = observation(i/10,features=dict(torso_angle_deg=72,head_y=.46,hip_y=.58,bbox_aspect=2.4),
                          posture_prediction=prediction('sitting'))
        events += det.process(obs)
    assert not [e for e in events if e.event_type=='possible_fall']
