"""Evaluation CLI.

  python -m hearth.eval batch [--seeds 5] [--scenarios ID ...] [--monitor module:factory]
                             [--monitor-config config.json] [--set floor_hold_seconds=2 ...]
      Run scenarios headless through a fresh monitor per run and score them. The default
      monitor is the real MonitorService with the shipped hearth/monitor/config.json.

  python -m hearth.eval study [--count 200] [--people 1-4] [--rooms day-room bedroom] [--fall-rate 0.6] [--sweep] [--jobs 8]
      Generate random scenarios (hearth.sim.generate), score them, and break the results
      down by fall kind, depth, people, activity and more. Writes recordings/eval/study/.

  python -m hearth.eval score --run recordings/sim/possible-fall-01-s42.run.json --events events.jsonl
  python -m hearth.eval score --start 2026-10-10T14:00:00Z --truth session.truth.jsonl --events export.jsonl --source jetson-01
      Score an event log captured elsewhere (the live WebSocket, or GET /events/export after a
      live session). Lines may be bare Events or ServerMessages; non-event messages are skipped.

  python -m hearth.eval real recordings/real/jetson [more clips or dirs] [--sweep floor_hip_y_min=0.4,0.45,0.5]
      Replay recorded clips (Jetson --record files, fall-model --record files, sim .stream.jsonl)
      through the real monitor and score each against its <clip>.truth.jsonl. Writes
      recordings/eval/real/. Unlabelled clips are replayed and their alerts listed.

  python -m hearth.eval timeline <clip.jsonl>
      Half-second timeline of a clip (posture state, features, failing floor rules, events),
      for timing truth segments by hand.

  python -m hearth.eval label-template <clip.jsonl | dir> [--force]
      Write <clip>.truth.jsonl: one whole-clip segment labelled from the clip name, to edit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from hearth_schema.models import Event, GroundTruth

from hearth.eval.batch import DEFAULT_MONITOR, DEFAULT_OUT_DIR, load_monitor_factory, run_batch, summary_markdown
from hearth.eval.real import DEFAULT_REAL_DIR, evaluate_recordings, sweep, sweep_markdown, timeline, timeline_text, write_results
from hearth.eval.real_monitor import CONFIG_ENV, OVERRIDES_ENV, load_config
from hearth.eval.recording import find_recordings, label_template, load_recording, truth_path, write_truth
from hearth.eval.scoring import EvalConfig, evaluate, summarize
from hearth.eval.study import DEFAULT_STUDY_DIR, build_report, plan_study, run_study, study_markdown, write_study
from hearth.sim.generate import ROOMS, GenSpec
from hearth.sim.generator import NoiseConfig
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


def _apply_monitor_config(args: argparse.Namespace) -> None:
    """Hand --monitor-config / --set to the real monitor (and to study worker processes)."""
    if getattr(args, "monitor_config", None):
        os.environ[CONFIG_ENV] = str(Path(args.monitor_config).resolve())
    overrides = {}
    for item in getattr(args, "set", None) or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        try:
            overrides[key] = json.loads(value)
        except ValueError:
            overrides[key] = value  # a bare string, e.g. --set posture_mode=model
    if overrides:
        os.environ[OVERRIDES_ENV] = json.dumps(overrides)
    if getattr(args, "monitor_config", None) or overrides:
        load_config()  # fail now on a bad file or key, not inside a worker


def _noise(args: argparse.Namespace) -> NoiseConfig | None:
    return NoiseConfig.edge_like() if getattr(args, "edge_like", False) else None


def _batch(args: argparse.Namespace) -> int:
    scenarios = [find_scenario(s) for s in args.scenarios] if args.scenarios else list(load_all().values())
    config = EvalConfig(alert_tolerance=args.tolerance)
    reports = run_batch(scenarios, args.seeds, load_monitor_factory(args.monitor), config, noise=_noise(args))
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
                   responder_rate=args.responder_rate, outage_rate=args.outage_rate,
                   collision_rate=args.collision_rate)  # fmt: skip
    plan = plan_study(base, count=args.count, people=_range(args.people), rooms=args.rooms, seed=args.seed,
                      sweep=args.sweep)  # fmt: skip
    print(f"Running {len(plan)} generated scenarios on {args.jobs} worker(s)...")
    outcomes = run_study(plan, args.monitor, EvalConfig(alert_tolerance=args.tolerance), jobs=args.jobs,
                         noise=_noise(args))
    out = Path(args.out)
    report = build_report(outcomes, args.monitor, worst=args.worst, reproduce_dir=out / "scenarios")
    write_study(report, outcomes, out)
    print(study_markdown(report))
    print(f"Study written to {out}")
    return 0


def _score(args: argparse.Namespace) -> int:
    if args.run:
        run_path = Path(args.run)
        meta = json.loads(run_path.read_text(encoding="utf-8"))
        truth_path = Path(args.truth) if args.truth else run_path.with_name(run_path.name.replace(".run.json", ".truth.jsonl"))
        start_text = meta["start"]
    else:
        if not args.truth:
            raise SystemExit("--start needs --truth (there is no run.json to find it next to)")
        truth_path, start_text = Path(args.truth), args.start
    truth = [GroundTruth.model_validate_json(line) for line in truth_path.read_text(encoding="utf-8").splitlines() if line]
    start = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
    report = evaluate(read_events(Path(args.events)), truth, start, source=args.source,
                      config=EvalConfig(alert_tolerance=args.tolerance))  # fmt: skip
    print(report.model_dump_json(indent=2))
    return 0


def _parse_value(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return text


def _real(args: argparse.Namespace) -> int:
    recs = [load_recording(p) for p in find_recordings(args.paths)]
    if not recs:
        raise SystemExit("no .jsonl clips found")
    config = EvalConfig(alert_tolerance=args.tolerance)
    results = evaluate_recordings(recs, load_monitor_factory(args.monitor), config)
    out = Path(args.out)
    markdown = write_results(results, out, args.monitor)
    for spec in args.sweep or []:
        key, sep, values = spec.partition("=")
        if not sep:
            raise SystemExit(f"--sweep expects key=v1,v2,..., got {spec!r}")
        rows = sweep(recs, key, [_parse_value(v) for v in values.split(",")], config)
        markdown += "\n" + sweep_markdown(key, rows)
    (out / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Results written to {out}")
    return 0


def _timeline(args: argparse.Namespace) -> int:
    rec = load_recording(args.clip)
    print(timeline_text(rec, timeline(rec, bucket_s=args.bucket)))
    return 0


def _label_template(args: argparse.Namespace) -> int:
    for path in find_recordings(args.paths):
        rec = load_recording(path)
        target = truth_path(path)
        if target.exists() and not args.force:
            print(f"skip {target} (exists; --force to overwrite)")
            continue
        write_truth(target, label_template(rec))
        print(f"wrote {target}: {', '.join(f'{g.truth_label} 0-{g.t_end:g}s' for g in label_template(rec))}")
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
    when = score.add_mutually_exclusive_group(required=True)
    when.add_argument("--run", help="<run>.run.json written by python -m hearth.sim run")
    when.add_argument("--start", help="ISO time of t=0 in the truth file (a live session); needs --truth")
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
    study.add_argument("--collision-rate", type=float, default=GenSpec().collision_rate)
    study.add_argument("--seed", type=int, default=0, help="first scenario seed")
    study.add_argument("--sweep", action="store_true", help="equal runs per people x room x fall kind")
    study.add_argument("--jobs", type=int, default=1, help="worker processes")
    study.add_argument("--worst", type=int, default=10, help="worst runs to list and save for replay")
    study.add_argument("--monitor", default=DEFAULT_MONITOR, help=f"monitor factory (default {DEFAULT_MONITOR})")
    study.add_argument("--tolerance", type=float, default=EvalConfig.alert_tolerance)
    study.add_argument("--out", default=DEFAULT_STUDY_DIR)

    for p in (batch, study):
        p.add_argument("--monitor-config", help="monitor config JSON (default hearth/monitor/config.json)")
        p.add_argument("--set", action="append", metavar="KEY=VALUE",
                       help="override one monitor config field, e.g. --set floor_hold_seconds=2 (repeatable)")
        p.add_argument("--edge-like", action="store_true",
                       help="emit observations shaped like the Jetson's (no keypoint confidence, face "
                            "checked every 0.5 s, brief dropouts skipped)")

    real = sub.add_parser("real", help="score recorded clips against their hand-labelled truth")
    real.add_argument("paths", nargs="+", help="clip .jsonl files and/or directories of them")
    real.add_argument("--monitor", default=DEFAULT_MONITOR, help=f"monitor factory (default {DEFAULT_MONITOR})")
    real.add_argument("--sweep", action="append", metavar="KEY=V1,V2,...",
                      help="also score the labelled clips once per value of a monitor setting (repeatable)")
    real.add_argument("--tolerance", type=float, default=EvalConfig.alert_tolerance)
    real.add_argument("--out", default=DEFAULT_REAL_DIR)

    timeline = sub.add_parser("timeline", help="print a clip's posture timeline, for labelling")
    timeline.add_argument("clip")
    timeline.add_argument("--bucket", type=float, default=0.5, help="seconds per row (default 0.5)")

    template = sub.add_parser("label-template", help="write a whole-clip <clip>.truth.jsonl to edit")
    template.add_argument("paths", nargs="+")
    template.add_argument("--force", action="store_true", help="overwrite existing truth files")

    for p in (real, timeline):
        p.add_argument("--monitor-config", help="monitor config JSON (default hearth/monitor/config.json)")
        p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override one monitor config field")

    args = parser.parse_args(argv)
    _apply_monitor_config(args)
    commands = {"batch": _batch, "study": _study, "score": _score, "real": _real, "timeline": _timeline,
                "label-template": _label_template}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
