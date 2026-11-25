"""The workspace the compiler reads.

Everything the compiler gets is here: the traces, the artifact contract, the
engine its package will run on, the runner's calling convention, and an empty
directory to write into. Nothing describes the target system, because the
trace is what describes it, and nothing describes the tasks that will be used to
judge the result.

The contract is materialised from the library rather than copied by hand, so the
schema the compiler validates against is the one the engine enforces.
"""

import json
import shutil
from pathlib import Path

from traceir.ir.models import graph_schema

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "docs/ir-v0.1.md"
LIBRARY = ROOT / "traceir"

# The deployment's calling convention. It describes who calls the workflow and
# with what, which is a fact about the runner; it says nothing about the task,
# and it deliberately does not mention the system the traces reached.
DEPLOYMENT = """# The runner

The workflow is executed by a runner, which supplies this task's record and
nothing else:

- `task_id`      the task's identifier
- `instruction`  what the user asked for, in their words
- `datetime`     the time the request is about
- `supervisor`   first_name, last_name, email, phone_number

`out/input.example.json` declares the input the workflow expects, and it is
filled from those fields.

When the workflow is done, it must have told the system so itself, as the traces
did. The runner ends the process, not the task: a terminal node is not a
completion, and the run's output is not a substitute for one.
"""


def build(destination: Path, traces: Path, task_ids: list[str]) -> Path:
    """Write a fresh workspace at `destination`, reading sessions from `traces`."""
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "ir").mkdir(parents=True)
    (destination / "out").mkdir()
    shutil.copy2(SPEC, destination / "ir/docs.md")
    (destination / "deployment.md").write_text(DEPLOYMENT)
    (destination / "ir/schema.json").write_text(
        json.dumps(graph_schema(), indent=2, allow_nan=False)
    )
    # The package is run inside the workspace, and the sandbox mounts nothing
    # from the repository, so the engine travels with it.
    shutil.copytree(
        LIBRARY, destination / "traceir", ignore=shutil.ignore_patterns("__pycache__")
    )
    for task_id in task_ids:
        session_directory = destination / "traces" / task_id
        session_directory.mkdir(parents=True)
        session = next((traces / task_id / "session").glob("*.jsonl"))
        shutil.copy2(session, session_directory / "session.jsonl")
    return destination
