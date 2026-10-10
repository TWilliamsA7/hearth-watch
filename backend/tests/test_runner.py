import json

from hearth.sim.__main__ import main
from hearth.sim.runner import run_headless, write_run
from hearth.sim.scenario import load_all
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
    assert meta["seed"] == 42 and meta["start"] == "2026-01-01T00:00:00Z"


def test_cli_runs_every_scenario(tmp_path, capsys):
    assert main(["run", "--all", "--out", str(tmp_path)]) == 0
    assert len(list(tmp_path.glob("*.stream.jsonl"))) == len(load_all())
    assert main(["run", "possible-fall-01", "--seed", "3", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "possible-fall-01-s3.truth.jsonl").exists()
