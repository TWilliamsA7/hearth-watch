import importlib.util
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hearth_schema import Event, GroundTruth, Heartbeat, Observation, PostureFeatures

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "shared/scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_contract_has_no_drift():
    schema_export = load_script("export_schema")
    for name, expected in schema_export.build_schemas().items():
        assert json.loads((ROOT / "shared/schema" / f"{name}.schema.json").read_text()) == expected
    types = load_script("export_types")
    bundle = json.loads((ROOT / "shared/schema/hearth.schema.json").read_text())
    assert (ROOT / "shared/types/hearth.ts").read_text() == types.render(bundle)


@pytest.mark.parametrize("name,model", [("observation", Observation), ("event", Event),
                                       ("heartbeat", Heartbeat), ("ground_truth", GroundTruth)])
def test_existing_fixtures_remain_compatible(name, model):
    model.model_validate_json((ROOT / "shared/fixtures" / f"{name}.example.json").read_text())


def test_nonfinite_features_rejected():
    with pytest.raises(ValidationError):
        PostureFeatures(vertical_velocity=float("nan"))


def test_type_generator_rejects_unsupported_shapes():
    types = load_script("export_types")
    with pytest.raises(ValueError):
        types.ts_type({"allOf": [{"type": "string"}]})


def test_frontend_types_match_the_schemas():
    """frontend/room-view/src/generated/*.ts come from the schema files; regenerate after changing them."""
    import shutil
    import subprocess

    room_view = ROOT / "frontend" / "room-view"
    node = shutil.which("node")
    if node is None or not (room_view / "node_modules" / "json-schema-to-typescript").exists():
        pytest.skip("node or frontend/room-view node_modules not installed")
    result = subprocess.run([node, "scripts/gen-types.mjs", "--check"], cwd=room_view,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + "\nrun: cd frontend/room-view && npm run gen:types"
