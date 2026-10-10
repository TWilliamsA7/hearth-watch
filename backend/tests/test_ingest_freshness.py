from fastapi.testclient import TestClient

from hearth.api.app import create_app
from conftest import observation, UPRIGHT


def edge(t, **changes):
    return observation(t, source_type="edge", **changes)


def test_old_queue_does_not_restore_offline_camera_or_recover_alert(service, clock):
    service.ingest([edge(i / 2) for i in range(7)])
    incident = next(iter(service.alerts.values()))
    clock.seconds = 10
    result = service.ingest([edge(4, features=UPRIGHT), edge(4.5, features=UPRIGHT)])
    assert result['stale'] == 2 and result['accepted'] == 0
    assert incident.recovered_at is None
    status = service.snapshot().sources[0]
    assert not status.online and status.observation_stale and status.posture == 'unknown'
    assert len([e for e in service.events if e.event_type == 'source_online']) == 1
    assert service.ingest([edge(10, features=UPRIGHT)])['accepted'] == 1
    assert incident.recovered_at is not None
    assert service.snapshot().sources[0].online


def test_future_timestamp_cannot_poison_ordering(service):
    assert service.ingest([edge(100)])['future'] == 1
    assert not service.observations and not service.health
    assert service.ingest([edge(0)])['accepted'] == 1


def test_mixed_batch_drops_old_frames_without_building_false_hold(service, clock):
    clock.seconds = 10
    result = service.ingest([edge(i / 2) for i in range(8)] + [edge(10)])
    assert result['stale'] == 8 and result['accepted'] == 1
    assert not service.alerts


def test_replay_and_simulation_clocks_are_not_wall_clock_filtered(service):
    for source_type in ('simulation', 'replay'):
        assert service.ingest([observation(-100, source_type=source_type)])['accepted'] == 1
        assert service.ingest([observation(100, source_type=source_type)])['accepted'] == 1


def test_configurable_age_boundaries_and_diagnostics(service, clock):
    service.config.edge_observation_max_age_seconds = 2
    service.config.edge_future_tolerance_seconds = 1
    clock.seconds = 10
    with TestClient(create_app(service=service)) as client:
        frames = [edge(7.9), edge(8), edge(11), edge(11.1)]
        body = {'observations': [o.model_dump(mode='json') for o in frames]}
        counts = client.post('/ingest/observations', json=body).json()
        assert counts == dict(accepted=2, stale=1, future=1, duplicates=0, out_of_order=0)
        status = client.get('/ingest/status').json()
        assert status['counts'] == counts
        assert status['edge_observation_max_age_seconds'] == 2
        assert status['scope'] == 'since_backend_start'
        client.post('/ingest/observations', json={'observations': [frames[1].model_dump(mode='json')]})
        assert client.get('/ingest/status').json()['counts']['duplicates'] == 1
