"""Evaluation CLI.

  python -m hearth.eval batch [--seeds 5] [--scenarios ID ...] [--monitor module:factory]
      Run scenarios headless through a fresh monitor per run and score them.

  python -m hearth.eval study [--count 200] [--people 1-4] [--rooms day-room bedroom] [--fall-rate 0.6] [--sweep] [--jobs 8]
      Generate random scenarios (hearth.sim.generate), score them, and break the results
      down by fall kind, depth, people, activity and more. Writes recordings/eval/study/.

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

from hearth.eval.batch import DEFAULT_MONITOR, DEFAULT_OUT_DIR, load_monitor_factory, run_batch, summary_markdown
from hearth.eval.scoring import EvalConfig, evaluate, summarize
from hearth.eval.study import DEFAULT_STUDY_DIR, build_report, plan_study, run_study, study_markdown, write_study
from hearth.sim.generate import ROOMS, GenSpec
from hearth.sim.scenario import find_scenario, load_all



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


def _range(text: str) -> tuple[int, int]:
    lo, _, hi = text.partition("-")
    return int(lo), int(hi or lo)


def _study(args: argparse.Namespace) -> int:
    base = GenSpec(duration=args.duration, fall_rate=args.fall_rate, recover_rate=args.recover_rate,
                   responder_rate=args.responder_rate, outage_rate=args.outage_rate)  # fmt: skip
    plan = plan_study(base, count=args.count, people=_range(args.people), rooms=args.rooms, seed=args.seed,
                      sweep=args.sweep)  # fmt: skip
    print(f"Running {len(plan)} generated scenarios on {args.jobs} worker(s)...")
    outcomes = run_study(plan, args.monitor, EvalConfig(alert_tolerance=args.tolerance), jobs=args.jobs)
    out = Path(args.out)
    report = build_report(outcomes, args.monitor, worst=args.worst, reproduce_dir=out / "scenarios")
    write_study(report, outcomes, out)
    print(study_markdown(report))
    print(f"Study written to {out}")
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

    study = sub.add_parser("study", help="generate many random scenarios, score them, break results down")
    study.add_argument("--count", type=int, default=100, help="runs (default 100)")
    study.add_argument("--people", default="1-4", help="people per scenario, e.g. 2 or 1-4 (default 1-4)")
    study.add_argument("--rooms", nargs="+", choices=sorted(ROOMS), default=sorted(ROOMS))
    study.add_argument("--duration", type=float, default=GenSpec().duration)
    study.add_argument("--fall-rate", type=float, default=0.6)
    study.add_argument("--recover-rate", type=float, default=GenSpec().recover_rate)
    study.add_argument("--responder-rate", type=float, default=GenSpec().responder_rate)
    study.add_argument("--outage-rate", type=float, default=GenSpec().outage_rate)
    study.add_argument("--seed", type=int, default=0, help="first scenario seed")
    study.add_argument("--sweep", action="store_true", help="equal runs per people x room x fall kind")
    study.add_argument("--jobs", type=int, default=1, help="worker processes")
    study.add_argument("--worst", type=int, default=10, help="worst runs to list and save for replay")
    study.add_argument("--monitor", default=DEFAULT_MONITOR, help=f"monitor factory (default {DEFAULT_MONITOR})")
    study.add_argument("--tolerance", type=float, default=EvalConfig.alert_tolerance)
    study.add_argument("--out", default=DEFAULT_STUDY_DIR)

    args = parser.parse_args(argv)
    return {"batch": _batch, "study": _study, "score": _score}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
