import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hearth.api.frontend import mount_frontend


def build(tmp_path):
    dashboard, studio, runs = tmp_path / "dist", tmp_path / "studio", tmp_path / "runs"
    (dashboard / "assets").mkdir(parents=True)
    (dashboard / "index.html").write_text("<html>dashboard</html>")
    (dashboard / "assets" / "app.js").write_text("app")
    (studio / "assets").mkdir(parents=True)
    (studio / "studio.html").write_text("<html>studio</html>")
    (studio / "index.html").write_text("<html>runs</html>")
    (studio / "assets" / "studio.js").write_text("studio")
    runs.mkdir()
    (runs / "fall-01-s1.run.json").write_text(json.dumps({"run_id": "fall-01-s1"}))
    (runs / "fall-01-s1.stream.jsonl").write_text('{"a":1}\n')
    (runs / "fall-01-s1.world.json").write_text('{"w":1}')
    (tmp_path / "secret.txt").write_text("secret")
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    mount_frontend(app, dashboard=dashboard, studio=studio, runs=runs)
    return TestClient(app)


def test_dashboard_pages_assets_and_api_share_one_port(tmp_path):
    client = build(tmp_path)
    for path in ("/", "/scenarios", "/recognition"):
        assert "dashboard" in client.get(path).text
    assert client.get("/assets/app.js").text == "app"
    assert client.get("/health").json() == {"status": "ok"}  # API routes still win
    assert client.get("/missing").status_code == 404
    assert client.get("/..%2Fsecret.txt").status_code == 404
    for old, new in {"/nurses": "/", "/room": "/", "/face-check-in": "/recognition"}.items():
        assert client.get(old, follow_redirects=False).headers["location"] == new


def test_studio_and_recorded_runs(tmp_path):
    client = build(tmp_path)
    assert client.get("/studio", follow_redirects=False).headers["location"] == "/studio/"
    assert "studio" in client.get("/studio/").text
    assert "runs" in client.get("/studio/runs").text
    assert client.get("/studio/assets/studio.js").text == "studio"
    assert client.get("/dev-api/runs").json() == [{"run_id": "fall-01-s1"}]
    assert client.get("/dev-api/runs/fall-01-s1/stream").text == '{"a":1}\n'
    assert client.get("/dev-api/runs/fall-01-s1/world").json() == {"w": 1}
    assert client.get("/dev-api/runs/fall-01-s1/truth").status_code == 404
    assert client.get("/dev-api/runs/..%2Fx/world").status_code == 404


def test_missing_build_explains_how_to_start(tmp_path):
    app = FastAPI()
    mount_frontend(app, dashboard=tmp_path / "none", studio=tmp_path / "none", runs=tmp_path / "none")
    client = TestClient(app)
    for path in ("/", "/studio/"):
        response = client.get(path)
        assert response.status_code == 503 and "start.ps1" in response.text
    assert client.get("/dev-api/runs").json() == []
