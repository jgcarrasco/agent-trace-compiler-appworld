"""Run a workflow package against one task and record what happened.

The engine runs in this process rather than through the CLI, because the record
wants the result object: the node trace, the state it accumulated, and the token
counts each model call reported. Grading belongs to the harness: the task's own
tests are run by the environment server afterwards, never by the package.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from traceir.engine.backend import DecisionBackend, GenerationBackend, Usage
from traceir.engine.cli import load_env_file
from traceir.engine.core import DEFAULT_MAX_STEPS, Completed
from traceir.engine.core import run as run_workflow
from traceir.engine.package import load_operations, load_workflow
from traceir.ir.validate import validate_graph

from bench.appworld.deciders import ChatDecider
from bench.appworld.serve import World

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "appworld_gen"

# What the harness can put in a workflow's initial input. A package that asks for
# anything else has baked one instance into its interface.
CANONICAL = ("task_id", "instruction", "datetime", "supervisor")


def initial_input(package: Path, task: dict) -> dict:
    """Read the shape the package declares and fill it from this task."""
    example_path = package / "input.example.json"
    if not example_path.is_file():
        raise ValueError(f"{example_path} is missing; the package must show the input it needs")
    example = json.loads(example_path.read_text())
    if not isinstance(example, dict):
        raise ValueError(f"{example_path} must hold an object, not {type(example).__name__}")
    unknown = sorted(set(example) - set(CANONICAL))
    if unknown:
        raise ValueError(
            f"the package asks for {unknown}, which the harness cannot supply "
            f"and which a workflow over new instances cannot need; it can fill {list(CANONICAL)}"
        )
    return {key: task[key] for key in example}


def backends(usage: Usage, decide_backend: str = "api"):
    """The two providers a workflow can reach, plus the DECIDE one asks for.

    `api` is the typed Decisions endpoint; `chat` is a large chat model given the
    same question and asked for one label, which is how the ladder's arm that
    separates the decision model from the decision's typing is run.
    """
    generate = GenerationBackend(
        url=os.environ["GENERATE_URL"],
        model=os.environ["GENERATE_MODEL"],
        api_key=os.environ["GENERATE_API_KEY"],
        usage=usage,
    )
    if decide_backend == "chat":
        decide = ChatDecider(
            url=os.environ["GENERATE_URL"],
            model=os.environ["GENERATE_MODEL"],
            api_key=os.environ["GENERATE_API_KEY"],
            usage=usage,
        )
    else:
        decide = DecisionBackend(
            url=os.environ["DECIDE_URL"],
            model=os.environ["DECIDE_MODEL"],
            api_key=os.environ["DECIDE_API_KEY"],
            usage=usage,
        )
    return generate, decide


def run_package(
    package: Path, task_id: str, world: World, *, max_steps: int, decide_backend: str = "api"
) -> tuple[dict, list, dict]:
    """Run one package on one task and return its record, usage and grading."""
    task = world.initialize(task_id)
    task["task_id"] = task_id
    # The package reaches the system through the deployment's own variables,
    # the same ones the trace agent had.
    os.environ["APPWORLD_ENV_URL"] = world.url
    os.environ["APPWORLD_TASK_ID"] = task_id

    graph, directory = load_workflow(package)
    operations, setup, teardown = load_operations(directory)
    workflow = validate_graph(graph, operations=set(operations))
    value = initial_input(package, task)

    usage = Usage()
    generate, decide = backends(usage, decide_backend)
    started = time.monotonic()
    result = run_workflow(
        workflow,
        value,
        operations=operations,
        generate=generate,
        decide=decide,
        max_steps=max_steps,
        setup=setup,
        teardown=teardown,
    )
    seconds = round(time.monotonic() - started, 1)
    evaluation = world.evaluate(task_id)

    completed = isinstance(result, Completed)
    record = {
        "task": task_id,
        "package": str(package),
        "completed": completed,
        "detail": None if completed else result.detail,
        "steps": len(result.trace),
        "node_trace": [[node, value] for node, value in result.trace],
        "state": result.state,
        "output": result.output if completed else None,
        "seconds": seconds,
        "max_steps": max_steps,
        "success": evaluation["success"],
        "tests_passed": evaluation["num_tests"] - len(evaluation["failures"]),
        "tests_total": evaluation["num_tests"],
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return record, usage.calls, evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", type=Path, required=True, help="the package's out/ directory")
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", type=Path, required=True, help="where the record goes")
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--decide-backend", choices=("api", "chat"), default="api",
                        help="who answers DECIDE: the typed Decisions endpoint, or a chat model")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    arguments = parser.parse_args()
    load_env_file(arguments.env_file)

    world = World(port=arguments.port, root=arguments.root).start()
    try:
        record, calls, evaluation = run_package(
            arguments.package, arguments.task, world, max_steps=arguments.max_steps,
            decide_backend=arguments.decide_backend,
        )
    finally:
        world.stop()

    arguments.out.mkdir(parents=True, exist_ok=True)
    completed = record["completed"]
    (arguments.out / "record.json").write_text(json.dumps(record, indent=2, default=str))
    (arguments.out / "usage.json").write_text(json.dumps(calls, indent=2, default=str))
    (arguments.out / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
    verdict = "success" if record["success"] else "FAILED"
    print(f"{arguments.task}: {verdict}, {record['tests_passed']}/{record['tests_total']} tests, "
          f"{record['steps']} steps, {len(calls)} model calls, {record['seconds']}s"
          + ("" if completed else f" (stopped: {record['detail'][:120]})"))


if __name__ == "__main__":
    main()
