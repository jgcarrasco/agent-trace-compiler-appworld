"""The independence rules that keep the IR and the engine reusable.

These are read from the source with `ast`, without importing or executing the
files under test:

* `traceir/ir` is the artifact contract, so it must not reach a model client, a
  benchmark, the trace launcher, or the engine it describes.
* `traceir/engine/core.py` is graph execution, so it carries the same
  restrictions plus the host layer that runs it.
* a workflow package is data plus plain functions, so it must not import the
  library that runs it.
* the published tool must not import the evaluation harness, which only exists
  for the experiments.

If a rule has to change, change the code and the rule in the same review.
"""

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]

# Model clients, benchmarks, and the trace launcher have no business in the
# contract or the engine.
HEAVY = ("httpx", "litellm", "openai", "appworld")

RULES = [
    ("traceir/ir", ROOT / "traceir/ir", (*HEAVY, "traceir.engine", "traceir.viz")),
    (
        "traceir/engine/core.py",
        ROOT / "traceir/engine/core.py",
        (*HEAVY, "traceir.engine", "traceir.compile"),
    ),
    # The compiler writes graphs; the contract and the executor must not need it.
    ("traceir/ir", ROOT / "traceir/ir", ("traceir.compile",)),
    (
        "traceir/engine/core.py",
        ROOT / "traceir/engine/core.py",
        ("traceir.compile",),
    ),
]

# The tool is what gets published; the harness only exists for the experiments.
TOOL = (ROOT / "traceir",)


def imported_names(path: Path) -> list[str]:
    """Top-level and nested import targets, taken from the source text."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def blocked(name: str, forbidden: tuple[str, ...]) -> bool:
    return any(name == entry or name.startswith(f"{entry}.") for entry in forbidden)


class PackageBoundaryTests(unittest.TestCase):
    def files(self, target: Path) -> list[Path]:
        return sorted(target.rglob("*.py")) if target.is_dir() else [target]

    def test_contract_and_engine_core_stay_lightweight(self):
        for label, target, forbidden in RULES:
            paths = self.files(target)
            self.assertTrue(paths, f"nothing to check for {label}")
            for path in paths:
                for name in imported_names(path):
                    with self.subTest(module=label, imported=name):
                        self.assertFalse(
                            blocked(name, forbidden),
                            f"{path.relative_to(ROOT)} imports {name!r}, which breaks "
                            f"what {label} is supposed to stay independent of",
                        )

    def test_the_published_tool_does_not_import_the_experiment_harness(self):
        for target in TOOL:
            for path in self.files(target):
                for name in imported_names(path):
                    with self.subTest(path=str(path.relative_to(ROOT)), imported=name):
                        self.assertFalse(
                            blocked(name, ("bench",)),
                            f"{path.relative_to(ROOT)} imports {name!r}: traceir must "
                            "stay usable without the evaluation harness",
                        )

    def test_workflow_packages_do_not_import_the_library(self):
        packages = sorted((ROOT / "workflows").rglob("*.py"))
        self.assertTrue(packages, "no workflow packages found")
        for path in packages:
            for name in imported_names(path):
                with self.subTest(module=str(path.relative_to(ROOT)), imported=name):
                    self.assertFalse(
                        blocked(name, ("traceir",)),
                        f"{path.relative_to(ROOT)} imports {name!r}: a workflow package is "
                        "graph data plus plain functions, and must not depend on its runner",
                    )

    def test_the_rules_cover_real_imports(self):
        # A silently vacuous guard is worse than none: these files do import
        # things, and the engine layer does reach the host and httpx layers.
        self.assertTrue(imported_names(ROOT / "traceir/engine/core.py"))
        self.assertTrue(blocked("httpx", HEAVY))
        self.assertTrue(
            blocked(
                "httpx",
                tuple(
                    name
                    for name in imported_names(ROOT / "traceir/engine/backend.py")
                    if name in HEAVY
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
