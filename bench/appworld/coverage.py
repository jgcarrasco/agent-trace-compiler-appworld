"""The coverage curve: how many traces does a family need before the workflow holds?

For each family and each k, the compiler is given the first k of a deterministic
sample of that family's traces, and the resulting package is run on every
held-out instance. The endpoint (all traces) reuses the compile already made for
the main experiment, so the curve ends where that experiment begins.

`--repeats` compiles the same family again from all its traces, which is how the
run-to-run spread of the compiler gets measured rather than described.

What this measures is not "can it compile" but "does it stop being one
instance's workflow": at k=1 the compiler has seen a single world, and the
interesting question is how many more it takes before the held-out set passes.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from traceir.engine.cli import load_env_file
from traceir.engine.core import DEFAULT_MAX_STEPS

from bench.appworld.run import run_package
from bench.appworld.serve import World

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "plans"
COVERAGE = ROOT / "runs/appworld/coverage"
STABILITY = ROOT / "runs/appworld/stability"
COMPILES = ROOT / "runs/appworld/compile"


def latest_compile(family: str) -> Path:
    found = sorted(COMPILES.glob(f"*/{family}/*/workspace/out"))
    if not found:
        raise SystemExit(f"no compile for {family}; run bench.appworld.compile first")
    return found[-1]


def compile_into(family: str, directory: Path, k: str | None) -> Path:
    """Compile into `directory`, or return the package that is already there."""
    package = directory / "workspace/out"
    if (directory / "compile.json").is_file():
        return package
    command = [sys.executable, "-m", "bench.appworld.compile",
               "--family", family, "--out", str(directory)]
    if k is not None:
        command += ["--k", k]
    subprocess.run(command, check=True, cwd=ROOT)
    return package


def evaluate(package: Path, family: str, base: Path, heldout: list[str], port: int) -> dict:
    """Run the package on every held-out instance and summarise the result."""
    base.mkdir(parents=True, exist_ok=True)
    if not (package / "graph.json").is_file():
        summary = {"family": family, "instances": len(heldout), "successes": 0,
                   "package": str(package), "no_graph": True}
        (base / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    successes, state_ok, passed, total = 0, 0, 0, 0
    with World(port=port) as world:
        for task in heldout:
            destination = base / "runs" / task
            if (destination / "record.json").is_file():
                record = json.loads((destination / "record.json").read_text())
                evaluation = json.loads((destination / "evaluation.json").read_text())
            else:
                try:
                    record, calls, evaluation = run_package(
                        package, task, world, max_steps=DEFAULT_MAX_STEPS
                    )
                except (ValueError, RuntimeError) as error:
                    record = {"task": task, "success": False, "steps": 0, "completed": False,
                              "detail": f"{type(error).__name__}: {error}",
                              "tests_passed": 0, "tests_total": 0}
                    calls, evaluation = [], {}
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "record.json").write_text(json.dumps(record, indent=2, default=str))
                (destination / "usage.json").write_text(json.dumps(calls, indent=2, default=str))
                (destination / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
            successes += record["success"]
            passed += record["tests_passed"]
            total += record["tests_total"]
            state_ok += only_the_answer_failed(evaluation)

    summary = {
        "family": family,
        "instances": len(heldout),
        "successes": successes,
        "state_ok": state_ok,
        "tests": f"{passed}/{total}",
        "package": str(package),
    }
    (base / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def only_the_answer_failed(evaluation: dict) -> bool:
    """Whether every requirement except the answer comparison passed.

    A workflow can make exactly the right changes and still fail the answer
    check, most often by never marking the task complete. That is a different
    outcome from making the wrong changes, and it is worth counting apart.
    """
    failures = evaluation.get("failures") or []
    return all(str(failure.get("requirement", "")).startswith("assert answers match") for failure in failures)


def show(label: str, summary: dict) -> None:
    if summary.get("no_graph"):
        print(f"{label}: the compiler produced no graph")
        return
    print(f"{label}: {summary['successes']}/{summary['instances']} held out, "
          f"state {summary['state_ok']}/{summary['instances']}, tests {summary['tests']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", default="p1")
    parser.add_argument("--config", default="pi-dsv41flash-high-v1")
    parser.add_argument("--families", default=None)
    parser.add_argument("--ks", default="1,3,5,10,all")
    parser.add_argument("--repeats", type=int, default=0,
                        help="extra compiles of the same family from all its traces")
    parser.add_argument("--port", type=int, default=9100)
    arguments = parser.parse_args()
    load_env_file(ROOT / ".env")

    plan = json.loads((PLANS / f"{arguments.plan}.json").read_text())
    families = plan["families"]
    if arguments.families:
        wanted = arguments.families.split(",")
        families = {name: value for name, value in families.items() if name in wanted}

    for family, entry in families.items():
        heldout = entry["heldout"]
        for k in [item for item in arguments.ks.split(",") if item]:
            base = COVERAGE / arguments.config / family / f"k{k}"
            package = latest_compile(family) if k == "all" else compile_into(family, base / "compile", k)
            show(f"{family} k={k:>3}", evaluate(package, family, base, heldout, arguments.port))
        for repeat in range(1, arguments.repeats + 1):
            base = STABILITY / arguments.config / family / f"run{repeat}"
            package = compile_into(family, base / "compile", None)
            show(f"{family} repeat {repeat}", evaluate(package, family, base, heldout, arguments.port))


if __name__ == "__main__":
    main()
