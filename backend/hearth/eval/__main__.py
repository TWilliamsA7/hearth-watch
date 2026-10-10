"""Evaluation CLI.

  python -m hearth.eval batch [--seeds 5] [--scenarios ID ...] [--monitor module:factory]
      Run scenarios headless through a fresh monitor per run and score them.

  python -m hearth.eval score --run recordings/sim/possible-fall-01-s42.run.json --events events.jsonl
      Score an event log captured elsewhere (e.g. from the live WebSocket during a simulated
      run). Lines may be bare Events or ServerMessages; non-event messages are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from hearth_schema.models import Event, GroundTruth

from hearth.eval.batch import DEFAULT_MONITOR, load_monitor_factory, run_batch, summary_markdown
from hearth.eval.scoring import EvalConfig, evaluate, summarize
from hearth.sim.scenario import find_scenario, load_all

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[3] / "recordings" / "eval"


def read_events(path: Path) -> list[Event]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        msg = json.loads(line)
        if "type" in msg and "data" in msg:
            if msg["type"] != "event":
                continue
            msg = msg["data"]
        events.append(Event.model_validate(msg))
    return events


def _batch(args: argparse.Namespace) -> int:
    scenarios = [find_scenario(s) for s in args.scenarios] if args.scenarios else list(load_all().values())
    config = EvalConfig(alert_tolerance=args.tolerance)
    reports = run_batch(scenarios, args.seeds, load_monitor_factory(args.monitor), config)
    summary = summarize(reports)

    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    for r in reports:
        (out / "runs" / f"{r.run_id}.report.json").write_text(r.model_dump_json(indent=2), encoding="utf-8")
    (out / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    markdown = summary_markdown(summary, reports, args.monitor)
    (out / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Reports written to {out}")
    return 0


def _score(args: argparse.Namespace) -> int:
    run_path = Path(args.run)
    meta = json.loads(run_path.read_text(encoding="utf-8"))
    truth_path = Path(args.truth) if args.truth else run_path.with_name(run_path.name.replace(".run.json", ".truth.jsonl"))
    truth = [GroundTruth.model_validate_json(line) for line in truth_path.read_text(encoding="utf-8").splitlines() if line]
    start = datetime.fromisoformat(meta["start"].replace("Z", "+00:00"))
    report = evaluate(read_events(Path(args.events)), truth, start, source=args.source,
                      config=EvalConfig(alert_tolerance=args.tolerance))  # fmt: skip
    print(report.model_dump_json(indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hearth.eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = parser.add_subparsers(dest="command", required=True)

    batch = sub.add_parser("batch", help="evaluate scenarios headless against the monitor")
    batch.add_argument("--scenarios", nargs="*", help="scenario ids or .yaml paths (default: all)")
    batch.add_argument("--seeds", type=int, default=5, help="seeds per scenario (default 5)")
    batch.add_argument("--monitor", default=DEFAULT_MONITOR, help=f"monitor factory (default {DEFAULT_MONITOR})")
    batch.add_argument("--tolerance", type=float, default=EvalConfig.alert_tolerance)
    batch.add_argument("--out", default=DEFAULT_OUT_DIR)

    score = sub.add_parser("score", help="score a captured event log against a run's ground truth")
    score.add_argument("--run", required=True, help="<run>.run.json written by python -m hearth.sim run")
    score.add_argument("--events", required=True, help="JSONL of Events or ServerMessages")
    score.add_argument("--truth", help="defaults to the .truth.jsonl next to --run")
    score.add_argument("--source", default="sim")
    score.add_argument("--tolerance", type=float, default=EvalConfig.alert_tolerance)

    args = parser.parse_args(argv)
    return _batch(args) if args.command == "batch" else _score(args)


if __name__ == "__main__":
    sys.exit(main())
