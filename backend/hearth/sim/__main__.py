"""Simulator CLI.

  python -m hearth.sim list
  python -m hearth.sim run possible-fall-01 [--seed 42] [--out DIR]
  python -m hearth.sim run --all
  python -m hearth.sim generate --people 3 --fall-rate 1 --seed 7 [--count 5] [--run]
  python -m hearth.sim generate --people 2 --fall-rate 0 --collision-rate 1 --run
  python -m hearth.sim live --people 6 --seed 3 --duration 300 [--fall-at 120] [--evaluate]
  python -m hearth.sim live --scenario live-fall-01 --evaluate
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from hearth.sim.generate import ROOMS, GenSpec, generate
from hearth.sim.generator import SIM_EPOCH, NoiseConfig
from hearth.sim.runner import DEFAULT_OUT_DIR, run_headless, write_run
from hearth.sim.scenario import DEFAULT_SCENARIOS_DIR, GENERATED_SUBDIR, find_scenario, load_all, save_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hearth.sim", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list available scenarios")

    run = sub.add_parser("run", help="generate scenarios headless and write JSONL")
    run.add_argument("scenarios", nargs="*", help="scenario ids or .yaml paths")
    run.add_argument("--all", action="store_true", help="run every scenario in scenarios/")
    run.add_argument("--seed", type=int, help="override the scenario's default seed")
    run.add_argument("--out", default=DEFAULT_OUT_DIR, help=f"output directory (default {DEFAULT_OUT_DIR})")
    run.add_argument("--now", action="store_true", help="timestamp from the current time instead of a fixed epoch")
    run.add_argument("--clean", action="store_true", help="no perception noise")

    gen = sub.add_parser("generate", help="generate random scenarios (YAML), and optionally run them")
    gen.add_argument("--people", type=int, default=GenSpec().people)
    gen.add_argument("--room", choices=sorted(ROOMS), default=GenSpec().room)
    gen.add_argument("--duration", type=float, default=GenSpec().duration, help="target seconds")
    gen.add_argument("--fall-rate", type=float, default=GenSpec().fall_rate, help="chance someone falls (0-1)")
    gen.add_argument("--recover-rate", type=float, default=GenSpec().recover_rate, help="chance a faller gets up")
    gen.add_argument("--responder-rate", type=float, default=GenSpec().responder_rate,
                     help="chance someone helps a faller who stays down")
    gen.add_argument("--outage-rate", type=float, default=GenSpec().outage_rate, help="chance of a camera outage")
    gen.add_argument("--collision-rate", type=float, default=GenSpec().collision_rate,
                     help="chance two people walk into each other (when nobody else falls)")
    gen.add_argument("--seed", type=int, default=0, help="first seed")
    gen.add_argument("--count", type=int, default=1, help="scenarios to generate, with consecutive seeds")
    gen.add_argument("--dir", default=DEFAULT_SCENARIOS_DIR / GENERATED_SUBDIR, help="where to write the YAML")
    gen.add_argument("--run", action="store_true", help="also run each one headless (for the room-view dev page)")
    gen.add_argument("--out", default=DEFAULT_OUT_DIR, help=f"run output directory (default {DEFAULT_OUT_DIR})")

    live = sub.add_parser("live", help="run the live engine headless and write the run (and optionally score it)")
    live.add_argument("--scenario", help="a scenarios/live/*.yaml id (other options then ignored)")
    live.add_argument("--people", type=int, default=4)
    live.add_argument("--seed", type=int, default=0)
    live.add_argument("--duration", type=float, default=300.0)
    live.add_argument("--width", type=float, help="room width, metres (default: from the seed)")
    live.add_argument("--depth", type=float, help="room depth, metres")
    live.add_argument("--camera", choices=["corner_left", "corner_right", "front", "ceiling"], default="corner_left")
    live.add_argument("--fall-at", type=float, action="append", default=[], help="schedule a fall at t (repeatable)")
    live.add_argument("--clean", action="store_true", help="no perception noise")
    live.add_argument("--evaluate", action="store_true", help="score the run against the real monitor")
    live.add_argument("--out", default=DEFAULT_OUT_DIR, help=f"output directory (default {DEFAULT_OUT_DIR})")

    args = parser.parse_args(argv)
    if args.command == "live":
        return _live(args)
    if args.command == "generate":
        spec = GenSpec(people=args.people, room=args.room, duration=args.duration, fall_rate=args.fall_rate,
                       recover_rate=args.recover_rate, responder_rate=args.responder_rate,
                       outage_rate=args.outage_rate, collision_rate=args.collision_rate)  # fmt: skip
        for seed in range(args.seed, args.seed + args.count):
            sc = generate(spec, seed)
            path = save_scenario(sc, Path(args.dir) / f"{sc.scenario_id}.yaml")
            print(f"{sc.scenario_id}: {sc.title} ({sc.duration:.1f}s) -> {path}")
            if args.run:
                result = run_headless(sc)
                print(f"  {result.name}: {len(result.observations)} observations -> {write_run(result, args.out)['stream']}")
        return 0
    if args.command == "list":
        for sc in load_all().values():
            print(f"{sc.scenario_id:24} {sc.duration:6.1f}s  seed {sc.seed:<4} {sc.title}")
        return 0

    if not args.all and not args.scenarios:
        parser.error("give scenario ids or --all")
    scenarios = list(load_all().values()) if args.all else [find_scenario(s) for s in args.scenarios]
    start = datetime.now(UTC).replace(microsecond=0) if args.now else SIM_EPOCH
    noise = NoiseConfig.off() if args.clean else None
    for sc in scenarios:
        result = run_headless(sc, args.seed, start=start, noise=noise)
        paths = write_run(result, args.out)
        print(f"{result.name}: {len(result.observations)} observations -> {paths['stream']}")
    return 0


def _live(args: argparse.Namespace) -> int:
    from hearth.sim.live.runner import LiveScenario, load_live_scenarios, run_live
    from hearth.sim.live.spec import CameraSpec, EngineSpec, IncidentSpec, RoomSpec

    if args.scenario:
        library = load_live_scenarios()
        if args.scenario not in library:
            print(f"no live scenario {args.scenario!r}; have: {', '.join(library) or 'none'}", file=sys.stderr)
            return 2
        sc = library[args.scenario]
    else:
        spec = EngineSpec(actors=args.people, room=RoomSpec(width=args.width, depth=args.depth),
                          camera=CameraSpec(preset=args.camera), noise=not args.clean,
                          incidents=[IncidentSpec(kind="fall", t=t, duration=40) for t in args.fall_at])  # fmt: skip
        sc = LiveScenario(title=f"Live engine, {args.people} residents", seed=args.seed, duration=args.duration, spec=spec)
    run = run_live(sc)
    paths = write_run(run, args.out)  # type: ignore[arg-type]
    print(f"{run.name}: {len(run.observations)} observations, {len(run.ground_truth)} truth segments -> {paths['stream']}")
    for rec in run.incidents:
        print(f"  {rec.kind} {rec.person_id} at {rec.t_start:.1f}s ({rec.motion or rec.source}), "
              f"floor {rec.t_floor}, recovered {rec.t_recovered}")  # fmt: skip
    if args.evaluate:
        from hearth.eval.batch import PerPersonMonitor, drive, load_monitor_factory
        from hearth.eval.scoring import EvalConfig, evaluate

        config = EvalConfig()
        events = drive(run, PerPersonMonitor(load_monitor_factory()),  # type: ignore[arg-type]
                       tail=max(config.alert_tolerance, config.check_tolerance) + 1.0)  # fmt: skip
        report = evaluate(events, run.ground_truth, run.start, config=config)
        print(f"  simulated results: {report.counts.model_dump()}")
        print(f"  detection delays (s): {report.detection_delays}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
