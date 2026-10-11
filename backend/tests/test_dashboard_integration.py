from datetime import timedelta
from fastapi.testclient import TestClient
from hearth.api.app import create_app
from conftest import BASE


def test_dashboard_routes_and_schedule(service):
    with TestClient(create_app(service=service), base_url='http://127.0.0.1:5173', client=('127.0.0.1', 12345)) as client:
        for path in ('/', '/scenarios', '/recognition'):
            # The built dashboard, or a page explaining how to build it.
            response = client.get(path, follow_redirects=False)
            assert response.status_code in {200, 503} and 'text/html' in response.headers['content-type']
        assert client.get('/face-check-in', follow_redirects=False).headers['location'] == '/recognition'
        assert client.get('/sim/live/status').status_code == 200
        response = client.post('/appointments', json=dict(resident_label='Demo Alex', room_id='room-01', scheduled_at=(BASE-timedelta(minutes=20)).isoformat(), demo=True))
        assert response.status_code == 201
        item = client.get('/appointments').json()[0]
        assert item['status'] == 'overdue'
        assert client.get('/alerts').json()[0]['kind'] == 'appointment_overdue'
        assert client.post(f"/appointments/{item['appointment_id']}/check-in", json={'method':'manual'}).json()['status'] == 'arrived'
        assert client.get('/face-demo/status', headers={'Origin':'http://127.0.0.1:5173'}).status_code == 200
        assert client.get('/face-demo/status', headers={'Origin':'http://example.com'}).status_code == 403
