"""Serve the built dashboard and the developer Sim Studio from the API's own port.

`scripts/start.ps1` builds both into `dist` folders. One origin keeps the
dashboard's WebSockets, the loopback-only recognition checks and the Jetson on
the same address. When a build is missing, pages explain how to make one.
"""
import json
from pathlib import Path
import re

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, Response

ROOT = Path(__file__).resolve().parents[3]
DASHBOARD = ROOT / "frontend" / "dist"
STUDIO = ROOT / "frontend" / "room-view" / "dist"
# Runs written by `python -m hearth.sim run` (git-ignored); the run player reads them.
RUNS = ROOT / "recordings" / "sim"
RUN_FILE = re.compile(r"^([a-z0-9-]+-s\d+)$")

# The dashboard picks its view from these paths; they all load the same page.
DASHBOARD_PAGES = ("/", "/room", "/nurses")

NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>Hearth · not built</title>
<body style="font-family:system-ui;max-width:40em;margin:4em auto;line-height:1.5">
<h1>The {name} has not been built</h1>
<p>Start Hearth with <code>.\\scripts\\start.ps1</code> from the repository root; it installs and
builds the interface before starting the server. The API itself is running: see <a href="/docs">/docs</a>.</p>"""


def _page(directory: Path, name: str, file: str = "index.html") -> Response:
    page = directory / file
    if not page.is_file():
        return HTMLResponse(NOT_BUILT.format(name=name), status_code=503)
    # The page names hashed bundles; never let a stale copy point at old ones.
    return FileResponse(page, headers={"Cache-Control": "no-store"})


def _file(directory: Path, relative: str) -> Response:
    base = directory.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path)


def create_runs_router(runs: Path = RUNS) -> APIRouter:
    """Recorded sim runs for the Studio's run player (was a Vite dev-server plugin)."""
    router = APIRouter(prefix="/dev-api/runs", tags=["developer"])

    @router.get("")
    @router.get("/")
    async def list_runs():
        files = sorted(runs.glob("*.run.json")) if runs.is_dir() else []
        return [json.loads(f.read_text(encoding="utf-8")) for f in files]

    @router.get("/{run_id}/{kind}")
    async def run_file(run_id: str, kind: str):
        if not RUN_FILE.match(run_id) or kind not in {"stream", "truth", "world"}:
            raise HTTPException(404, "Not found")
        world = kind == "world"  # one JSON document; the others are JSON lines
        path = runs / f"{run_id}.{kind}.{'json' if world else 'jsonl'}"
        if not path.is_file():
            raise HTTPException(404, "Run not found")
        return PlainTextResponse(path.read_text(encoding="utf-8"),
                                 media_type="application/json" if world else "application/x-ndjson")

    return router


def mount_frontend(app: FastAPI, *, dashboard: Path = DASHBOARD, studio: Path = STUDIO, runs: Path = RUNS) -> None:
    """Call last: the file route below matches any path no API route claimed."""
    app.include_router(create_runs_router(runs))

    for path in DASHBOARD_PAGES:
        app.add_api_route(path, lambda: _page(dashboard, "dashboard"), methods=["GET"], include_in_schema=False)

    @app.get("/face-check-in", include_in_schema=False)
    async def face_check_in():
        # Enrollment now lives in the dashboard's patient view.
        return RedirectResponse("/room#recognition")

    @app.get("/studio", include_in_schema=False)
    async def studio_redirect():
        return RedirectResponse("/studio/")

    @app.get("/studio/", include_in_schema=False)
    async def studio_page():
        return _page(studio, "Sim Studio", "studio.html")

    @app.get("/studio/runs", include_in_schema=False)
    async def run_player():
        return _page(studio, "run player")

    @app.get("/studio/{path:path}", include_in_schema=False)
    async def studio_file(path: str):
        return _file(studio, path)

    @app.get("/{path:path}", include_in_schema=False)
    async def dashboard_file(path: str):
        return _file(dashboard, path)
