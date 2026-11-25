#!/usr/bin/env python
"""Run an IR v0.1 workflow from the command line.

    uv run python -m traceir.engine --workflow workflows/email_reply

This file is the host for command-line use: it loads a workflow package, chooses
how the two roles are answered, and prints the trace, the state, and the output.
A benchmark host does the same for its own environment and calls the engine
directly; only the engine is shared, which is why nothing here belongs in it.

A workflow package is a directory holding `graph.json`, an optional
`operations.py` defining `WORKFLOW_OPERATIONS`, and an optional
`input.example.json` used as the default initial input. The package supplies its
own operations, including any that speak to a person, so it is code for the
target it was written against: the example's `request_input` prints and reads a
line, which is what a terminal can do. A host may still replace any of them by
name here.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Callable

from traceir.engine.backend import (
    DebugBackend,
    DecisionBackend,
    GenerationBackend,
    Usage,
)
from traceir.engine.core import DEFAULT_MAX_STEPS, Failed
from traceir.engine.core import run as run_workflow
from traceir.engine.package import load_operations, load_workflow
from traceir.ir.models import Workflow
from traceir.ir.validate import validate_graph

# Variables read from the environment, or from --env-file when unset.
GENERATE_URL = "GENERATE_URL"
GENERATE_MODEL = "GENERATE_MODEL"
GENERATE_KEY = "GENERATE_API_KEY"
DECIDE_URL = "DECIDE_URL"
DECIDE_MODEL = "DECIDE_MODEL"
DECIDE_KEY = "DECIDE_API_KEY"


def preview(value: object, limit: int = 70) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "..."


def run(
    workflow: Workflow,
    operations: dict,
    initial_input: dict,
    *,
    generate,
    decide,
    max_steps: int,
    setup: Callable | None = None,
    teardown: Callable | None = None,
) -> int:
    """Execute the workflow and print what it did, what it knew, and what it gave."""
    result = run_workflow(
        workflow,
        initial_input,
        operations=operations,
        generate=generate,
        decide=decide,
        max_steps=max_steps,
        setup=setup,
        teardown=teardown,
    )
    if result.trace:
        print("Trace:")
        for node_id, value in result.trace:
            print(f"  {node_id} -> {preview(value)}")
    if isinstance(result, Failed):
        print(f"\nFailed: {result.detail}", file=sys.stderr)
        exit_code = 1
    else:
        print(f"\nCompleted after {len(result.trace)} steps.")
        print(f"Output:\n{json.dumps(result.output, indent=2)}")
        exit_code = 0
    # Printed on failure too: what the workflow had established matters most then.
    print(f"\nState:\n{json.dumps(result.state, indent=2)}")
    return exit_code


def load_env_file(path: Path) -> None:
    """Fill unset variables from a `KEY=VALUE` file, so a local .env is enough.

    Variables already in the environment win, so exporting one is still the way
    to override what the file says.
    """
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip("\"'").strip())


def require_env(name: str, env_file: Path) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is not set; export it or add it to {env_file}")
    return value


def build_backends(arguments, usage: Usage | None = None):
    """Resolve the two role backends, creating a client only where one is used.

    Each role is answered by a model or by whoever is running this, and the two
    roles have nothing in common beyond that: GENERATE needs a chat model on an
    OpenAI-compatible endpoint, DECIDE needs a decision model on the Decisions
    API, so they have separate endpoints, keys, and models. The recorder, when
    given, collects what each model-backed call reports about itself.
    """
    load_env_file(arguments.env_file)
    debug = DebugBackend()

    generate = debug.generate
    if arguments.generate_backend == "chat":
        generate = GenerationBackend(
            url=require_env(GENERATE_URL, arguments.env_file),
            model=require_env(GENERATE_MODEL, arguments.env_file),
            api_key=require_env(GENERATE_KEY, arguments.env_file),
            max_tokens=arguments.max_tokens,
            usage=usage,
        )

    decide = debug.decide
    if arguments.decide_backend == "decisions":
        decide = DecisionBackend(
            url=require_env(DECIDE_URL, arguments.env_file),
            model=require_env(DECIDE_MODEL, arguments.env_file),
            api_key=require_env(DECIDE_KEY, arguments.env_file),
            usage=usage,
        )

    return generate, decide


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an IR v0.1 workflow from the command line.",
        prog="python -m traceir.engine",
    )
    parser.add_argument(
        "--workflow", type=Path, required=True,
        help="workflow directory or graph.json path",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="initial input JSON (default: <workflow>/input.example.json)",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--generate-backend", choices=("debug", "chat"), default="debug",
        help="how GENERATE nodes get their value (default: ask at the terminal)",
    )
    parser.add_argument(
        "--decide-backend", choices=("debug", "decisions"), default="debug",
        help="how DECIDE nodes get their answer (default: ask at the terminal)",
    )
    parser.add_argument(
        "--env-file", type=Path, default=Path(".env"),
        help="KEY=VALUE file read for variables not already exported; "
             ".env.example lists the six variables it can hold",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="output token cap per GENERATE call",
    )
    parser.add_argument(
        "--usage-out", type=Path, default=None,
        help="write one record per model-backed call, as the provider reported it",
    )
    arguments = parser.parse_args(argv)

    usage = Usage()
    try:
        generate, decide = build_backends(arguments, usage)
    except ValueError as error:
        print(f"Invalid backend configuration: {error}", file=sys.stderr)
        return 1

    try:
        graph, directory = load_workflow(arguments.workflow)
        operations, setup, teardown = load_operations(directory)
        workflow = validate_graph(graph, operations=set(operations))
        input_path = arguments.input or directory / "input.example.json"
        if not input_path.is_file():
            raise ValueError(f"no initial input at {input_path}; pass --input")
        initial_input = json.loads(input_path.read_text())
    except (OSError, ValueError) as error:
        print(f"Invalid workflow: {error}", file=sys.stderr)
        return 1

    try:
        exit_code = run(
            workflow,
            operations,
            initial_input,
            generate=generate,
            decide=decide,
            max_steps=arguments.max_steps,
            setup=setup,
            teardown=teardown,
        )
    except ValueError as error:
        # The engine raises only for a caller mistake, such as input that does
        # not match the graph: everything that happens during the run is reported
        # as a Failed result instead.
        print(f"Invalid input: {error}", file=sys.stderr)
        return 1
    # Written on failure too: a run that stopped early still cost what it cost.
    if arguments.usage_out:
        arguments.usage_out.write_text(json.dumps(usage.calls, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
