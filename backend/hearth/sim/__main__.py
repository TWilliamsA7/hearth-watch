"""Simulator CLI.

  python -m hearth.sim list
  python -m hearth.sim run possible-fall-01 [--seed 42] [--out DIR]
  python -m hearth.sim run --all
  python -m hearth.sim generate --people 3 --fall-rate 1 --seed 7 [--count 5] [--run]
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
    gen.add_argument("--seed", type=int, default=0, help="first seed")
    gen.add_argument("--count", type=int, default=1, help="scenarios to generate, with consecutive seeds")
    gen.add_argument("--dir", default=DEFAULT_SCENARIOS_DIR / GENERATED_SUBDIR, help="where to write the YAML")
    gen.add_argument("--run", action="store_true", help="also run each one headless (for the room-view dev page)")
    gen.add_argument("--out", default=DEFAULT_OUT_DIR, help=f"run output directory (default {DEFAULT_OUT_DIR})")

    args = parser.parse_args(argv)
    if args.command == "generate":
        spec = GenSpec(people=args.people, room=args.room, duration=args.duration, fall_rate=args.fall_rate,
                       recover_rate=args.recover_rate, responder_rate=args.responder_rate,
                       outage_rate=args.outage_rate)  # fmt: skip
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


if __name__ == "__main__":
    sys.exit(main())
