"""Loading a workflow package from its directory.

A package is a directory holding `graph.json`, an optional `operations.py`
defining `WORKFLOW_OPERATIONS`, and an optional `input.example.json`. Loading it
is the same work whatever runs it, so it lives here rather than in a runner: the
CLI reads packages this way, and so does a host that drives the engine in-process
and needs the result object rather than printed output.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Callable

from traceir.ir.validate import load_graph


def load_workflow(path: Path) -> tuple[dict, Path]:
    """Return the graph and its package directory, from a directory or file."""
    graph_path = path / "graph.json" if path.is_dir() else path
    if not graph_path.is_file():
        raise ValueError(f"no graph found at {graph_path}")
    return load_graph(graph_path), graph_path.parent


def load_operations(directory: Path) -> tuple[dict, Callable | None, Callable | None]:
    """Import `<directory>/operations.py` and return its operations and hooks.

    `setup(initial_input)` and `teardown()` are optional: a package that needs no
    prepared resource defines neither and behaves as before.

    The directory goes on `sys.path` first, so a package may keep part of its code
    in a sibling module, and it stays there because an operation may import at call
    time rather than at load time.
    """
    module_path = directory / "operations.py"
    if not module_path.is_file():
        return {}, None, None
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("workflow_operations", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    operations = getattr(module, "WORKFLOW_OPERATIONS", None)
    if not isinstance(operations, dict):
        raise ValueError(f"{module_path} defines no WORKFLOW_OPERATIONS mapping")
    setup = getattr(module, "setup", None)
    teardown = getattr(module, "teardown", None)
    return dict(operations), setup, teardown
