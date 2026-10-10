"""Invariant 1: ground truth never reaches the monitor.

Nothing under hearth/monitor may import the simulator, the evaluator, or GroundTruth.
"""

import ast
from pathlib import Path

MONITOR = Path(__file__).resolve().parents[1] / "hearth" / "monitor"
FORBIDDEN_MODULES = ("hearth.sim", "hearth.eval")


def _violations(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
        if isinstance(node, ast.Import):
            modules = [(a.name, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [(node.module, f"{node.module}.{a.name}") for a in node.names]
        else:
            if isinstance(node, ast.Name) and node.id == "GroundTruth":
                found.append(f"{path.name}:{node.lineno} uses GroundTruth")
            continue
        for module, full in modules:
            if module.startswith(FORBIDDEN_MODULES) or full.endswith(".GroundTruth"):
                found.append(f"{path.name}:{node.lineno} imports {full}")
    return found


def test_monitor_never_touches_ground_truth():
    problems = [v for path in sorted(MONITOR.rglob("*.py")) for v in _violations(path)]
    assert not problems, "\n".join(problems)


def test_checker_catches_violations(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("from hearth.sim.scenario import load_all\nfrom hearth_schema.models import GroundTruth\n")
    assert len(_violations(bad)) == 2
