"""Write JSON Schema files for the shared contract into shared/schema/.

Run from the repo root after changing shared/python/hearth_schema/models.py,
in any Python 3.11+ environment with the package installed:
    pip install -e shared/python
    python shared/scripts/export_schema.py
"""

import json
from pathlib import Path

from pydantic import TypeAdapter
from pydantic.json_schema import models_json_schema

from hearth_schema import (
    Alert,
    RoomSnapshot,
    Event,
    GroundTruth,
    Heartbeat,
    Observation,
    ObservationBatch,
    ServerMessage,
    SourceStatus,
)

OUT = Path(__file__).resolve().parents[1] / "schema"

STANDALONE = {
    "alert": Alert,
    "room_snapshot": RoomSnapshot,
    "observation": Observation,
    "event": Event,
    "ground_truth": GroundTruth,
    "heartbeat": Heartbeat,
    "observation_batch": ObservationBatch,
    "source_status": SourceStatus,
}


def write(name: str, schema: dict) -> None:
    path = OUT / f"{name}.schema.json"
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(OUT.parents[1])}")


def build_schemas() -> dict[str, dict]:
    schemas = {name: model.model_json_schema(mode="serialization")
               for name, model in STANDALONE.items()}
    schemas["server_message"] = TypeAdapter(ServerMessage).json_schema(mode="serialization")

    # One bundle with every model under $defs; the frontend generates its types from this.
    _, bundle = models_json_schema(
        [(m, "serialization") for m in STANDALONE.values()],
        title="Hearth contract v1",
    )
    msg = TypeAdapter(ServerMessage).json_schema(mode="serialization", ref_template="#/$defs/{model}")
    bundle["$defs"].update(msg.pop("$defs", {}))
    bundle["$defs"]["ServerMessage"] = msg
    schemas["hearth"] = bundle
    return schemas


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, schema in build_schemas().items():
        write(name, schema)


if __name__ == "__main__":
    main()
