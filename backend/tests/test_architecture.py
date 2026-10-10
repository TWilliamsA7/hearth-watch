"""Prevent explicit imports and dynamic references to privileged simulation data."""
import ast
from pathlib import Path

MONITOR = Path(__file__).resolve().parents[1] / "hearth/monitor"


def forbidden_references(source):
    violations = []
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [alias.name for alias in node.names]
        elif isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Also catches importlib.import_module('hearth.sim') / getattr(..., 'GroundTruth').
            names = [node.value]
        for name in names:
            parts = name.split(".")
            if any(part in {"sim", "eval", "GroundTruth"} for part in parts):
                violations.append((node.lineno, name))
    return violations


def test_monitor_never_imports_simulator_evaluator_or_ground_truth():
    failures = {str(path): forbidden_references(path.read_text(encoding="utf-8"))
                for path in MONITOR.rglob("*.py")}
    assert not {path: violations for path, violations in failures.items() if violations}


def test_guard_catches_absolute_relative_and_dynamic_imports():
    for source in ["import hearth.sim.generator", "from hearth import sim", "from ..eval import metrics",
                   "from hearth_schema import GroundTruth as Truth", "schema.GroundTruth",
                   "importlib.import_module('hearth.sim')"]:
        assert forbidden_references(source), source
