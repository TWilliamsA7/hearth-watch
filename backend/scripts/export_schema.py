"""Write JSON Schema for backend-only API shapes into backend/schema/.

The shared contract (shared/schema/) covers records that cross every boundary. These are
the extra shapes the dashboard reads from the simulator and evaluator: scene geometry,
run metadata, scenario control status, the 3D world track, evaluation reports and the live engine's
spec, frames and actor details. Frontend TypeScript types are generated from both.

    python backend/scripts/export_schema.py
"""

import json
from pathlib import Path

from hearth.eval.report import EvaluationReport, EvaluationSummary, StudyReport
from hearth.sim.controller import ScenarioInfo, SimStatus, TruthNow
from hearth.sim.live.incidents import IncidentRecord
from hearth.sim.live.models import ActorDetail, LiveFrame, LiveStatus, LiveWorld
from hearth.sim.live.spec import EngineSpec, IncidentSpec
from hearth.sim.runner import RunInfo
from hearth.sim.scenario import Scene
from hearth.sim.world import WorldTrack
from pydantic.json_schema import models_json_schema

OUT = Path(__file__).resolve().parents[1] / "schema" / "sim.schema.json"
MODELS = [Scene, RunInfo, ScenarioInfo, SimStatus, TruthNow, WorldTrack, EvaluationReport, EvaluationSummary, StudyReport,
          EngineSpec, IncidentSpec, IncidentRecord, LiveWorld, LiveFrame, LiveStatus, ActorDetail]


def main() -> None:
    _, bundle = models_json_schema([(m, "serialization") for m in MODELS], title="Hearth simulator and evaluation API v1")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(OUT.parents[2])}")


if __name__ == "__main__":
    main()
