"""Simulator CLI.

  python -m hearth.sim list
  python -m hearth.sim run possible-fall-01 [--seed 42] [--out DIR]
  python -m hearth.sim run --all
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from hearth.sim.generator import SIM_EPOCH, NoiseConfig
from hearth.sim.runner import DEFAULT_OUT_DIR, run_headless, write_run
from hearth.sim.scenario import find_scenario, load_all


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

    args = parser.parse_args(argv)
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
