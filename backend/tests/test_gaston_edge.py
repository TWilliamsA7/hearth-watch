"""Wire fixtures produced with origin/gaston/edge 9d9ccba, not fabricated model output."""
import json
from pathlib import Path

from fastapi.testclient import TestClient
from hearth_schema import Heartbeat, Observation
from hearth.api.app import create_app
from hearth.monitor.config import Zone
from conftest import observation


def fixture():
    return json.loads((Path(__file__).parent / 'fixtures/gaston_edge.json').read_text())


def test_fixture_matches_what_the_edge_sends_today():
    keypoints = [k for o in fixture()['observations'] for k in o.get('keypoints') or []]
    assert keypoints and all('confidence' not in k for k in keypoints)  # poseNet reports none


def test_gaston_batch_heartbeat_face_and_pose_feed(service):
    wire = fixture()
    with TestClient(create_app(service=service)) as client:
        assert client.post('/ingest/heartbeat', json=wire['heartbeat']).status_code == 200
        with client.websocket_connect('/ws/room/room-01') as ws:
            # Heartbeat snapshot: online event and source status.
            for _ in range(2):
                ws.receive_json()
            result = client.post('/ingest/observations', json={'observations': wire['observations']})
            assert result.json()['accepted'] == 9
            assert ws.receive_json()['data']['posture'] == 'standing'
        snap = client.get('/rooms/room-01/snapshot').json()
        assert snap['sources'][0]['posture'] == 'standing'
        assert snap['sources'][0]['posture_model'] is None
        assert snap['sources'][0]['fps'] == 12.7
        kinds = [e['event_type'] for e in snap['events']]
        assert kinds.count('possible_fall') == kinds.count('recovered') == 1
        assert kinds.count('face_present') == kinds.count('face_absent') == 1
        assert snap['alerts'][0]['recovered_at'] and snap['alerts'][0]['status'] == 'open'
        assert client.post('/ingest/observations', json={'observations': wire['observations']}).json()['duplicates'] == 9


def test_pose_status_stale_despite_heartbeat_and_offline(service, clock):
    wire = fixture()
    service.ingest([Observation.model_validate(wire['observations'][0])])
    for second in (1, 2, 3):
        clock.seconds = second
        service.heartbeat(Heartbeat.model_validate(wire['heartbeat']))
    status = service.snapshot().sources[0]
    assert status.online and status.observation_stale and status.posture == 'unknown'
    clock.seconds = 6
    service.check_health()
    assert not status.online
    assert [a.kind for a in service.alerts.values()] == ['source_offline']


def test_optional_confidence_does_not_authorize_gesture_or_zone(service):
    service.config.zones = [Zone(name='bed', polygon=[(0,0),(1,0),(1,1),(0,1)])]
    points = [dict(name='nose', x=.5, y=.3), dict(name='left_shoulder', x=.4, y=.45),
              dict(name='left_wrist', x=.4, y=.1), dict(name='left_hip', x=.5, y=.85)]
    service.ingest([observation(i/2, keypoints=points) for i in range(8)])
    assert len(service.alerts) == 1
    alert = next(iter(service.alerts.values()))
    assert alert.kind == 'possible_fall' and alert.severity == 'warning'


def test_gaston_replay_without_run_id_is_separate_from_live(service):
    live = Observation.model_validate(fixture()['observations'][0])
    replay = live.model_copy(update={'source': 'replay-01', 'source_type': 'replay'})
    assert service.ingest([live, replay])['accepted'] == 2
    assert {s.source_type for s in service.snapshot().sources} == {'edge', 'replay'}
