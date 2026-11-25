"""Compile one family's traces into a workflow package.

The compiler is the same agent as the tracer, without its tool: it reads the
workspace, writes a package into `out/`, and runs what it wrote against a live
world. That world holds this family's traced instances and nothing else, so the
instances used to judge the result do not exist anywhere it can reach.

The workspace is rebuilt for every compile, as is the world under it: two
compiles should differ because of the compiler, not because of what a previous
one left behind.
"""

import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from traceir.compile.run_agent import run_agent
from traceir.engine.cli import load_env_file

from bench.appworld import workspace as workspace_module
from bench.appworld.serve import DATA_ROOT, World

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "plans"
TRACES = ROOT / "runs/appworld/traces"
COMPILES = ROOT / "runs/appworld/compile"
SHARED = ROOT / "appworld_data/data"


def build_world(run_directory: Path, trace_directory: Path, task_ids: list[str]) -> Path:
    """A root holding only the traced instances of this family, and nothing else."""
    root = run_directory / "world"
    tasks = root / "data/tasks"
    tasks.mkdir(parents=True)
    for task_id in task_ids:
        shutil.copytree(
            DATA_ROOT / "data/tasks" / task_id,
            tasks / task_id,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    for name in ("base_dbs", "api_docs", "datasets", "version.txt"):
        (root / "data" / name).symlink_to(SHARED / name)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", default="p1")
    parser.add_argument("--family", required=True)
    parser.add_argument("--config", default="pi-dsv41flash-high-v1")
    parser.add_argument("--instance", default=None, help="task to leave initialized (default: first trace)")
    parser.add_argument("--k", type=int, default=None,
                        help="how many traces to give the compiler (default: all of them)")
    parser.add_argument("--seed", type=int, default=0, help="which k traces are chosen")
    parser.add_argument("--out", type=Path, default=None, help="exact directory for this compile")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    arguments = parser.parse_args()
    load_env_file(arguments.env_file)

    json.loads((PLANS / f"{arguments.plan}.json").read_text())
    trace_directory = TRACES / arguments.config / arguments.family
    task_ids = sorted(path.name for path in trace_directory.iterdir() if path.is_dir())
    if not task_ids:
        raise SystemExit(f"no traces under {trace_directory}")
    if arguments.k:
        # Deterministic subset: the curve is only readable if k means the same
        # thing every time it is compiled.
        chosen = random.Random(arguments.seed).sample(task_ids, min(arguments.k, len(task_ids)))
        task_ids = sorted(chosen)
    instance = arguments.instance or task_ids[0]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = arguments.out or (COMPILES / arguments.config / arguments.family / stamp)
    if arguments.out and run_directory.exists():
        raise SystemExit(f"{run_directory} already exists; remove it to compile again")
    run_directory.mkdir(parents=True)
    world_root = build_world(run_directory, trace_directory, task_ids)
    workspace = workspace_module.build(run_directory / "workspace", trace_directory, task_ids)
    print(f"{arguments.family}: {len(task_ids)} traces, world initialized on {instance}")

    started = time.monotonic()
    with World(port=arguments.port, root=world_root) as world:
        world.initialize(instance)
        attempt = run_agent(
            workspace,
            model=os.environ["COMPILER_MODEL"],
            run_dir=run_directory / "run",
            env={"APPWORLD_ENV_URL": world.url, "APPWORLD_TASK_ID": instance},
            timeout=arguments.timeout,
        )
    package = workspace / "out"
    produced = sorted(path.name for path in package.iterdir()) if package.is_dir() else []
    summary = {
        "family": arguments.family,
        "instance": instance,
        "plan": arguments.plan,
        "config": arguments.config,
        "stamp": stamp,
        "traces": len(task_ids),
        "trace_ids": task_ids,
        "model": os.environ["COMPILER_MODEL"],
        "seconds": attempt.seconds,
        "exit_code": attempt.exit_code,
        "produced": produced,
        "has_graph": "graph.json" in produced,
        "has_operations": "operations.py" in produced,
        "total_seconds": round(time.monotonic() - started, 1),
    }
    (run_directory / "compile.json").write_text(json.dumps(summary, indent=2))
    print(f"  exit {attempt.exit_code} in {attempt.seconds:.0f}s, out/: {produced or 'empty'}")
    print(f"  {run_directory}")


if __name__ == "__main__":
    main()
