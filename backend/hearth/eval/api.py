"""HTTP routes for evaluation and after-action data (/eval). See docs/interfaces.md section 3.4.

    GET /eval/runs            live simulated runs completed this session
    GET /eval/runs/{run_id}   EvaluationReport for one run (ground truth vs monitor events)
    GET /eval/summary         latest batch summary (python -m hearth.eval batch)

All results are simulated; any UI showing them must say so.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from hearth.eval.batch import DEFAULT_OUT_DIR
from hearth.eval.report import EvaluationReport, EvaluationSummary
from hearth.sim.controller import SimController
from hearth.sim.runner import RunInfo


def create_eval_router(controller: SimController, batch_dir: Path = DEFAULT_OUT_DIR) -> APIRouter:
    router = APIRouter(tags=["evaluation"])

    @router.get("/runs", response_model=list[RunInfo])
    async def runs() -> list[RunInfo]:
        return controller.runs()

    @router.get("/runs/{run_id}", response_model=EvaluationReport)
    async def run_report(run_id: str) -> EvaluationReport:
        try:
            return controller.report(run_id)
        except KeyError:
            raise HTTPException(404, f"no finished run {run_id!r}") from None

    @router.get("/summary", response_model=EvaluationSummary)
    async def summary() -> EvaluationSummary:
        path = Path(batch_dir) / "summary.json"
        if not path.exists():
            raise HTTPException(404, "no batch results yet; run: python -m hearth.eval batch")
        return EvaluationSummary.model_validate_json(path.read_text(encoding="utf-8"))

    return router
