"""Compare the arms on the held-out instances.

Everything here is computed from what the runs recorded: the agents' own token
counts, the workflow's per-call usage, and the price table in `bench/pricing.py`.
Nothing is priced at run time, so a price correction re-derives the numbers
instead of invalidating the runs.

Two tables come out of it. The first is per family, with the compile cost and the
break-even that belongs to it. The second is the ladder: the same held-out
instances, one row per arm, which is what separates the two changes the main
comparison makes at once.
"""

import argparse
import glob
import json
import statistics
from pathlib import Path

from bench.pricing import cost, fingerprint

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/appworld/run"
BASELINE = ROOT / "runs/appworld/baseline"
COMPILES = ROOT / "runs/appworld/compile"
TRACES = ROOT / "runs/appworld/traces"
PLANS = ROOT / "plans"

# Where each arm keeps what it recorded.
AGENT = BASELINE / "pi-dsv41flash-high-v1"
CHAT_DECIDE = BASELINE / "decide-chat"
ALL_GENERATE = BASELINE / "all-generate"


def usage_tokens(usage: list[dict]) -> tuple[int, int, int]:
    """Input, cached-input and output tokens a run's calls reported.

    Tokens are the quantity that does not depend on a price list or on how well
    a prompt happens to cache, so they are reported next to the dollars.
    """
    tok_in = tok_cached = tok_out = 0
    for call in usage:
        report = call.get("usage") or {}
        if "input_tokens" in report:
            tok_in += report["input_tokens"]
            tok_out += report.get("output_tokens", 0)
        else:
            tok_in += report.get("prompt_tokens", 0)
            tok_out += report.get("completion_tokens", 0)
            tok_cached += report.get("prompt_cache_hit_tokens", 0)
    return tok_in, tok_cached, tok_out


def usage_cost(usage: list[dict], *, cached: bool = True) -> float:
    """Price one run from the calls it recorded.

    The two endpoints report their tokens under different names, and the report
    says which one answered: the Decisions API counts `input_tokens`, an
    OpenAI-compatible endpoint counts `prompt_tokens`, cache included.

    With `cached=False` every input token is priced at the miss rate, which is
    what the run would have cost with no prefix cache. Which of the two is
    reported matters: caching moves these arms by very different factors.
    """
    total = 0.0
    for call in usage:
        report = call.get("usage") or {}
        if "input_tokens" in report:
            total += cost(
                call["model"],
                input_tokens=report["input_tokens"],
                output_tokens=report.get("output_tokens", 0),
            )
        else:
            total += cost(
                call["model"],
                input_tokens=report.get("prompt_tokens", 0),
                output_tokens=report.get("completion_tokens", 0),
                cache_hit_tokens=report.get("prompt_cache_hit_tokens", 0) if cached else 0,
            )
    return total


def agent_cost(meta: dict, *, cached: bool = True) -> float:
    """The counting agent's per-turn usage, priced as chat calls."""
    if cached:
        return cost(
            "deepseek-flash",
            input_tokens=meta["input"],
            output_tokens=meta["output"],
            cache_hit_tokens=meta["cacheRead"],
        )
    return cost(
        "deepseek-flash",
        input_tokens=meta["input"] + meta["cacheRead"],
        output_tokens=meta["output"],
    )


def only_the_answer_failed(path: Path) -> bool:
    """Whether every requirement except the answer comparison passed.

    A workflow can make exactly the right changes and never mark the task
    complete; that is a different outcome from making the wrong changes, so it
    gets its own column.
    """
    evaluation = json.loads(path.read_text())
    failures = evaluation.get("failures") or []
    return all(str(failure.get("requirement", "")).startswith("assert answers match")
               for failure in failures)


def workflow_run(directory: Path) -> dict:
    """One run of a workflow package, however its DECIDE nodes were answered."""
    record = json.loads((directory / "record.json").read_text())
    usage = json.loads((directory / "usage.json").read_text())
    return {
        "task": record["task"],
        "success": record["success"],
        "state_ok": only_the_answer_failed(directory / "evaluation.json"),
        "tests_passed": record["tests_passed"],
        "tests_total": record["tests_total"],
        "cost": usage_cost(usage),
        "cost_uncached": usage_cost(usage, cached=False),
        "tokens": usage_tokens(usage),
        "seconds": record["seconds"],
        "calls": len(usage),
        "steps": record.get("steps"),
        "max_steps": record.get("max_steps"),
    }


def generate_run(directory: Path) -> dict:
    """One run of the all-generate arm: a single call, priced from its usage."""
    record = json.loads((directory / "record.json").read_text())
    usage = json.loads((directory / "usage.json").read_text())
    return {
        "task": record["task"],
        "success": record["success"],
        "state_ok": only_the_answer_failed(directory / "evaluation.json"),
        "tests_passed": record["tests_passed"],
        "tests_total": record["tests_total"],
        "cost": usage_cost(usage),
        "cost_uncached": usage_cost(usage, cached=False),
        "tokens": usage_tokens(usage),
        "seconds": record["seconds"],
        "calls": len(usage),
        "steps": None,
        "max_steps": None,
    }


def agent_run(directory: Path) -> dict:
    """One native agent run, priced from what its session reported."""
    meta = json.loads((directory / "meta.json").read_text())
    return {
        "task": meta["task_id"],
        "success": meta["success"],
        "state_ok": only_the_answer_failed(directory / "evaluation.json"),
        "tests_passed": meta["tests_passed"],
        "tests_total": meta["tests_total"],
        "cost": agent_cost(meta),
        "cost_uncached": agent_cost(meta, cached=False),
        "tokens": (meta["input"] + meta["cacheRead"], meta["cacheRead"], meta["output"]),
        "seconds": meta["seconds"],
        "calls": meta["calls"],
        "steps": None,
        "max_steps": None,
    }


def family_runs(store: Path, family: str, read, *, experiment: str | None = None,
                subdir: str = "") -> list[dict]:
    """Every run directory of one arm on one family, read by that arm's reader."""
    base = store / experiment / family if experiment else store / family
    return [read(Path(path)) for path in sorted(glob.glob(str(base / "*" / subdir)))]


def trace_cost(family: str, config: str = "pi-dsv41flash-high-v1") -> float:
    """What it cost to collect one family's traces, at the recorded prices."""
    total = 0.0
    for directory in sorted((TRACES / config / family).glob("*")):
        total += agent_cost(json.loads((directory / "meta.json").read_text()))
    return total


def compile_cost(family: str) -> tuple[float, float]:
    """What the newest compile of this family cost, and how long it took."""
    sessions = sorted(glob.glob(str(COMPILES / "*" / family / "*" / "run/session/*.jsonl")))
    if not sessions:
        return 0.0, 0.0
    total = 0.0
    for line in Path(sessions[-1]).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        message = record.get("message") if record.get("type") == "message" else None
        if not message or message.get("role") != "assistant":
            continue
        usage = message.get("usage", {})
        total += cost(
            "deepseek-flash",
            input_tokens=usage.get("input", 0),
            output_tokens=usage.get("output", 0),
            cache_hit_tokens=usage.get("cacheRead", 0),
        )
    summary = json.loads((Path(sessions[-1]).parents[2] / "compile.json").read_text())
    return total, summary["seconds"]


CELLS = (f"{'n':>3} {'solved':>7} {'state ok':>9} {'tests':>9} {'tok in':>8} "
         f"{'$/run':>10} {'$ uncached':>11} {'s/run':>7} {'calls':>6}")


def cells(runs: list[dict]) -> str:
    """The columns every arm shares: how much it solved and what it cost.

    A dollar figure is a mean over runs rather than a total, because the arms
    ran on the same instances and the per-run price is the number that scales.
    """
    if not runs:
        return f"{'-':>3}"
    passed = sum(run["tests_passed"] for run in runs)
    total = sum(run["tests_total"] for run in runs)
    tests = str(passed) + "/" + str(total)
    return (
        f"{len(runs):>3} {sum(run['success'] for run in runs):>7} "
        f"{sum(run['state_ok'] for run in runs):>9} {tests:>9} "
        f"{statistics.mean(run['tokens'][0] for run in runs):>8,.0f} "
        f"{statistics.mean(run['cost'] for run in runs):>10.6f} "
        f"{statistics.mean(run['cost_uncached'] for run in runs):>11.6f} "
        f"{statistics.median(run['seconds'] for run in runs):>7.1f} "
        f"{statistics.mean(run['calls'] for run in runs):>6.1f}"
    )


def ladder(plan: dict, experiment: str) -> None:
    """The four arms side by side on the same held-out instances."""
    print("ladder\n")
    print(f"{'arm':22} {CELLS}")
    print("-" * (22 + len(CELLS) + 1))
    arms = [
        ("IR", lambda family: family_runs(RUNS, family, workflow_run, experiment=experiment,
                                          subdir="ir")),
        ("IR + chat decisions", lambda family: family_runs(CHAT_DECIDE, family, workflow_run)),
        ("agent", lambda family: family_runs(AGENT, family, agent_run)),
        ("all-generate", lambda family: family_runs(ALL_GENERATE, family, generate_run)),
    ]
    for label, read in arms:
        runs = [run for family in plan["families"] for run in read(family)]
        print(f"{label:22} {cells(runs)}")
        print(f"{'':22} spent ${sum(run['cost'] for run in runs):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", default="p1")
    arguments = parser.parse_args()
    plan = json.loads((PLANS / f"{arguments.experiment}.json").read_text())
    print(f"prices: {fingerprint()[:12]}\n")

    header = f"{'family':10} {'arm':6} {CELLS}"
    print(header)
    print("-" * len(header))
    totals = {"ir": [0, 0, 0.0], "agent": [0, 0, 0.0]}
    for family in plan["families"]:
        runs = family_runs(RUNS, family, workflow_run, experiment=arguments.experiment,
                          subdir="ir")
        costs = [run["cost"] for run in runs]
        truncated = [run["task"] for run in runs if run["steps"] is not None
                     and run["steps"] >= (run["max_steps"] or 1 << 30)]
        if runs:
            print(f"{family:10} {'IR':6} {cells(runs)}")
            totals["ir"][0] += sum(run["success"] for run in runs)
            totals["ir"][1] += len(runs)
            totals["ir"][2] += sum(costs)
            if truncated:
                print(f"{family:10} !! step limit hit on {len(truncated)} run(s): "
                      f"{', '.join(truncated)}")

        agents = family_runs(AGENT, family, agent_run)
        if agents:
            agent_costs = [run["cost"] for run in agents]
            print(f"{family:10} {'agent':6} {cells(agents)}")
            totals["agent"][0] += sum(run["success"] for run in agents)
            totals["agent"][1] += len(agents)
            totals["agent"][2] += sum(agent_costs)

        compile_dollars, compile_seconds = compile_cost(family)
        saving = statistics.mean(agent_costs) - statistics.mean(costs) if agents and runs else 0
        break_even = compile_dollars / saving if saving > 0 else float("inf")
        print(f"{'':10} compile ${compile_dollars:.4f}, {compile_seconds / 60:.1f} min, "
              f"break-even after {break_even:.0f} runs")
        print()

    print(f"{'pooled':10} {'IR':6} {totals['ir'][1]:>3} {totals['ir'][0]:>8} "
          f"{totals['ir'][2] / max(totals['ir'][1], 1):>10.6f}")
    print(f"{'pooled':10} {'agent':6} {totals['agent'][1]:>3} {totals['agent'][0]:>8} "
          f"{totals['agent'][2] / max(totals['agent'][1], 1):>10.6f}")
    print()
    ladder(plan, arguments.experiment)


if __name__ == "__main__":
    main()
