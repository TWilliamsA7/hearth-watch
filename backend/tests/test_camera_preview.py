from fastapi import FastAPI
from fastapi.testclient import TestClient
from hearth.api.camera_preview import create_preview_router


def test_preview_source_isolation_expiry_and_clear(service):
    now = [0.]
    app = FastAPI(); app.include_router(create_preview_router(service, clock=lambda: now[0]))
    with TestClient(app,base_url='http://127.0.0.1:5173',client=('127.0.0.1',12345)) as client:
        path = '/camera-preview/room-01/camera-a'
        headers = {'Content-Type':'image/jpeg','Origin':'http://127.0.0.1:5173'}
        assert client.post(path,content=b'\xff\xd8example\xff\xd9',headers=headers).status_code == 200
        response = client.get(path)
        assert response.content == b'\xff\xd8example\xff\xd9'
        assert response.headers['cache-control'] == 'no-store'
        assert client.get('/camera-preview/room-01/camera-b').status_code == 503
        assert client.get(path,headers={'Origin':'http://other.test'}).status_code == 403
        now[0] = 4
        assert client.get(path).status_code == 503
        assert client.post(path,content=b'bad',headers=headers).status_code == 422
        assert client.post(path,content=b'\xff\xd8example\xff\xd9',headers=headers).status_code == 200
        assert client.delete(path).status_code == 200
        assert client.get(path).status_code == 503
    with TestClient(app,base_url='http://localhost',client=('192.168.1.2',12345)) as client:
        assert client.get(path).status_code == 403
