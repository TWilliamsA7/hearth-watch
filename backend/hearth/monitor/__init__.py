"""Monitoring consumes observations only; no simulator or evaluator dependencies."""


def create_monitor():
    """Factory used by headless evaluation; contains no ground-truth dependencies."""
    from .adapter import MonitorAdapter
    from .config import MonitorConfig
    import os
    return MonitorAdapter(MonitorConfig.load(os.environ.get("HEARTH_CONFIG")))
