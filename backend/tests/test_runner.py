import json

from hearth.sim.__main__ import main
from hearth.sim.runner import run_headless, write_run
from hearth.sim.scenario import load_all
from hearth.sim.world import WorldTrack
from hearth_schema.models import GroundTruth, Heartbeat, Observation


def test_written_run_round_trips_through_the_contract(tmp_path):
    run = run_headless(load_all()["possible-fall-01"])
    paths = write_run(run, tmp_path)

    lines = [json.loads(line) for line in paths["stream"].read_text().splitlines()]
    parsed = [(Observation if m["type"] == "observation" else Heartbeat).model_validate(m["data"]) for m in lines]
    assert parsed == run.items

    truth = [GroundTruth.model_validate_json(line) for line in paths["truth"].read_text().splitlines()]
    assert truth == run.ground_truth

    meta = json.loads(paths["run"].read_text())
    assert meta["seed"] == 42 and meta["start"] == "2026-01-01T00:00:00Z" and meta["run_id"] == "possible-fall-01-s42"

    world = WorldTrack.model_validate_json(paths["world"].read_text())
    assert world == run.world() and world.scenario_id == "possible-fall-01"


def test_cli_runs_every_scenario(tmp_path, capsys):
    assert main(["run", "--all", "--out", str(tmp_path)]) == 0
    assert len(list(tmp_path.glob("*.stream.jsonl"))) == len(load_all())
    assert main(["run", "possible-fall-01", "--seed", "3", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "possible-fall-01-s3.truth.jsonl").exists()


def test_exported_api_schema_is_current():
    """backend/schema/sim.schema.json feeds the frontend types; regenerate it when models change."""
    import subprocess
    import sys
    from pathlib import Path

    schema = Path(__file__).resolve().parents[1] / "schema" / "sim.schema.json"
    before = schema.read_text(encoding="utf-8")
    subprocess.run([sys.executable, str(schema.parents[1] / "scripts" / "export_schema.py")], check=True,
                   capture_output=True)  # fmt: skip
    assert schema.read_text(encoding="utf-8") == before, "run python backend/scripts/export_schema.py and commit"
