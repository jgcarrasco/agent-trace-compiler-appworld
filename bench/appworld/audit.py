"""Recompute every number the paper reports, in the order the paper uses them.

The manuscript quotes numbers from the records under ``runs/``. This script
prints them, labelled, so the output can be read against the PDF line by line.
The default mode compares the output with the committed ``audit.txt`` and fails
when they differ, which is what happens when a record, a price, or the code that
reads them changes.

    uv run python -m bench.appworld.audit           # print, and compare
    uv run python -m bench.appworld.audit --write    # accept current numbers
    uv run python -m bench.appworld.audit --print    # print only
"""

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

from bench.appworld import analyze as A
from bench.pricing import cost, fingerprint, READ_ON

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "runs/appworld/traces/pi-dsv41flash-high-v1"
COVERAGE = ROOT / "runs/appworld/coverage/pi-dsv41flash-high-v1"
STABILITY = ROOT / "runs/appworld/stability/pi-dsv41flash-high-v1"
AUDIT = ROOT / "bench/appworld/audit.txt"

CONFIG = "pi-dsv41flash-high-v1"
K_VALUES = ("1", "3", "5", "10", "all")


class Report:
    """Lines of `label   value`, grouped, with the width fixed so it diffs."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures: list[str] = []

    def section(self, title: str) -> None:
        self.lines.append("")
        self.lines.append(f"== {title}")

    def item(self, label: str, value: object, width: int = 46) -> None:
        self.lines.append(f"{label:<{width}}{value}")

    def check(self, label: str, ok: bool, detail: str = "") -> None:
        self.item(label, ("yes " if ok else "NO  ") + detail)
        if not ok:
            self.failures.append(label)

    def text(self) -> str:
        return "\n".join(self.lines).strip() + "\n"


def mean(values) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def ratio(numerator: float, denominator: float) -> str:
    return f"{numerator / denominator:.3g}x" if denominator else "-"


def trace_runs(family: str) -> list[dict]:
    """The trace agent's records: one per solved instance."""
    runs = []
    for path in sorted(glob.glob(str(TRACES / family / "*"))):
        directory = Path(path)
        meta = json.loads((directory / "meta.json").read_text())
        evaluation = json.loads((directory / "evaluation.json").read_text())
        runs.append({"task": directory.name, "success": bool(evaluation.get("success")),
                     "meta": meta, "evaluation": evaluation})
    return runs


def package(family: str) -> dict:
    """The compiled graph of the package the main results use."""
    sessions = sorted(glob.glob(str(A.COMPILES / "*" / family / "*" / "workspace/out/graph.json")))
    return json.loads(Path(sessions[-1]).read_text()) if sessions else {}


def instances(report: Report, plan: dict) -> None:
    report.section("instances")
    total = held = 0
    dev_instructions: dict[str, set] = {}
    for family, entry in plan["families"].items():
        report.item(f"{family}  generated / dev / held out",
                    f"{len(entry['instances'])} / {len(entry['dev_pool'])} / "
                    f"{len(entry['heldout'])}, {entry['instruction_count']} instructions")
        total += len(entry["instances"])
        held += len(entry["heldout"])
        instruction = {one["task_id"]: one["instruction"] for one in entry["instances"]}
        dev_instructions[family] = {instruction[task] for task in entry["dev_pool"]}
    report.item("total generated / held out", f"{total} / {held}")

    covered, disjoint = True, True
    for family, entry in plan["families"].items():
        instruction = {one["task_id"]: one["instruction"] for one in entry["instances"]}
        for task in entry["heldout"]:
            if instruction[task] not in dev_instructions[family]:
                covered = False
        if set(entry["dev_pool"]) & set(entry["heldout"]):
            disjoint = False
    report.check("every held-out instruction appears in a trace", covered)
    report.check("traced and held-out instances are disjoint", disjoint)


def traces(report: Report, plan: dict) -> None:
    report.section("traces (the compiler's input)")
    calls_all, seconds_all, hopeless, traces_all = [], [], 0, []
    spent = 0.0
    for family in plan["families"]:
        runs = trace_runs(family)
        calls = [run["meta"]["calls"] for run in runs]
        seconds = [run["meta"]["seconds"] for run in runs]
        spent += sum(A.agent_cost(run["meta"]) for run in runs)
        traces_all += runs
        calls_all += calls
        seconds_all += seconds
        hopeless += sum(1 for run in runs if not run["success"])
        report.item(f"{family}  traces / solved",
                    f"{len(runs)} / {sum(r['success'] for r in runs)}")
    endings = 0
    for family in plan["families"]:
        for directory in sorted(glob.glob(str(TRACES / family / "*"))):
            for path in glob.glob(f"{directory}/session/*.jsonl"):
                code = ""
                for line in Path(path).read_text().splitlines():
                    record = json.loads(line) if line.strip() else {}
                    message = record.get("message") if record.get("type") == "message" else None
                    if not message or message.get("role") != "assistant":
                        continue
                    for block in message.get("content", []):
                        if block.get("type") == "toolCall" and block.get("name") == "run":
                            code = block.get("arguments", {}).get("code", "")
                endings += "complete_task" in code
    report.item("traces ending in the completion call", f"{endings} of {len(traces_all)}")
    report.item("calls per trace, min / median / max",
                f"{min(calls_all)} / {statistics.median(calls_all):.0f} / {max(calls_all)}")
    report.item("wall clock per trace, median s", f"{statistics.median(seconds_all):.0f}")
    report.item("total wall clock, min", f"{sum(seconds_all) / 60:.0f}")
    report.item("trace cost, $", f"{spent:.3f}")
    report.check("every trace solved its task", hopeless == 0, f"({hopeless} failures)")


def compiles(report: Report, plan: dict) -> None:
    report.section("compiles (the workflow's input-output contract)")
    for family in plan["families"]:
        graph = package(family)
        nodes = graph.get("nodes", [])
        categories = [node["type"] for node in nodes]
        cash, seconds = A.compile_cost(family)
        report.item(f"{family}  newest compile, $ / min", f"{cash:.3f} / {seconds / 60:.1f}")
        # A DECIDE node carries criteria rather than an operation, so the
        # distinct-operation count is over the nodes that name one.
        operations = {node["operation"] for node in nodes if "operation" in node}
        report.item(f"{family}  nodes  EXECUTE / DECIDE / GENERATE",
                    f"{len(nodes)}  {categories.count('EXECUTE')} / "
                    f"{categories.count('DECIDE')} / {categories.count('GENERATE')}")
        report.item(f"{family}  distinct operations", len(operations))


def main_runs(report: Report, plan: dict, experiment: str) -> None:
    report.section("main results (30 held out)")
    total: dict[str, list] = {"ir": [], "agent": []}
    for family in plan["families"]:
        for label in ("IR", "agent"):
            if label == "IR":
                runs = A.family_runs(A.RUNS, family, A.workflow_run,
                                     experiment=experiment, subdir="ir")
            else:
                runs = A.family_runs(A.AGENT, family, A.agent_run)
            total[label.lower() if label != "IR" else "ir"] += runs
            report.item(f"{family} {label}  solved / state / tests",
                        f"{sum(r['success'] for r in runs)}/{len(runs)}  "
                        f"{sum(r['state_ok'] for r in runs)}  "
                        f"{sum(r['tests_passed'] for r in runs)}/"
                        f"{sum(r['tests_total'] for r in runs)}")
            report.item(f"{family} {label}  $/run, s/run, calls",
                        f"{mean(r['cost'] for r in runs):.6f}  "
                        f"{statistics.median(r['seconds'] for r in runs):.1f}  "
                        f"{mean(r['calls'] for r in runs):.1f}")
    for label, runs in total.items():
        report.item(f"pooled {label}  solved / tests",
                    f"{sum(r['success'] for r in runs)}/{len(runs)}  "
                    f"{sum(r['tests_passed'] for r in runs)}/"
                    f"{sum(r['tests_total'] for r in runs)}")
        report.item(f"pooled {label}  $/run, s/run, calls",
                    f"{mean(r['cost'] for r in runs):.6f}  "
                    f"{statistics.median(r['seconds'] for r in runs):.1f}  "
                    f"{mean(r['calls'] for r in runs):.1f}")
        report.item(f"pooled {label}  $/run uncached",
                    f"{mean(r['cost_uncached'] for r in runs):.6f}")
        report.item(f"pooled {label}  tokens in / cached / out",
                    "  ".join(f"{mean(r['tokens'][i] for r in runs):,.0f}" for i in range(3)))
    lost = [run for run in total["agent"] if not run["success"]]
    for run in lost:
        report.item(f"agent's failing run: {run['task']}  tests / state",
                    f"{run['tests_passed']}/{run['tests_total']}  {run['state_ok']}")

    ir, agent = total["ir"], total["agent"]
    for family in plan["families"]:
        package_cash, _ = A.compile_cost(family)
        ir_runs = A.family_runs(A.RUNS, family, A.workflow_run, experiment=experiment,
                                subdir="ir")
        agent_runs = A.family_runs(A.AGENT, family, A.agent_run)
        saving = mean(r["cost"] for r in agent_runs) - mean(r["cost"] for r in ir_runs)
        traces_cash = A.trace_cost(family)
        report.item(f"{family} compile / traces, $",
                    f"{package_cash:.4f} / {traces_cash:.4f}")
        report.item(f"{family} break-even, executions", f"{package_cash / saving:.1f}")
        report.item(f"{family} break-even with traces, executions",
                    f"{(package_cash + traces_cash) / saving:.1f}")
    for label, runs in (("agent", agent), ("IR", ir)):
        cache_read = mean(r["tokens"][1] for r in runs)
        report.item(f"pooled {label}  cache read share of input",
                    f"{cache_read / mean(r['tokens'][0] for r in runs):.3f}")
    report.item("IR is cheaper than the agent by (mean)",
                ratio(mean(r["cost"] for r in agent), mean(r["cost"] for r in ir)))
    report.item("IR is cheaper than the agent by (median)",
                ratio(statistics.median(r["cost"] for r in agent),
                      statistics.median(r["cost"] for r in ir)))
    report.item("IR is faster than the agent by",
                ratio(statistics.median(r["seconds"] for r in agent),
                      statistics.median(r["seconds"] for r in ir)))
    report.item("IR uses fewer input tokens by",
                ratio(mean(r["tokens"][0] for r in agent), mean(r["tokens"][0] for r in ir)))


def ladder(report: Report, plan: dict, experiment: str) -> None:
    report.section("ladder (same 30 held out)")
    arms = {
        "all-generate": lambda family: A.family_runs(
            A.ALL_GENERATE, family, A.generate_run),
        "agent": lambda family: A.family_runs(A.AGENT, family, A.agent_run),
        "IR + chat decisions": lambda family: A.family_runs(
            A.CHAT_DECIDE, family, A.workflow_run),
        "IR": lambda family: A.family_runs(
            A.RUNS, family, A.workflow_run, experiment=experiment, subdir="ir"),
    }
    figures = {}
    for label, read in arms.items():
        runs = [run for family in plan["families"] for run in read(family)]
        figures[label] = (mean(r["cost"] for r in runs), mean(r["cost_uncached"] for r in runs),
                          mean(r["tokens"][0] for r in runs),
                          mean(r["cost_uncached"] for r in runs) / mean(r["cost"] for r in runs))
        report.item(f"{label}  solved / tests",
                    f"{sum(r['success'] for r in runs)}/{len(runs)}  "
                    f"{sum(r['tests_passed'] for r in runs)}/"
                    f"{sum(r['tests_total'] for r in runs)}")
        report.item(f"{label}  $/run, $ uncached, tok in",
                    f"{figures[label][0]:.6f}  {figures[label][1]:.6f}  "
                    f"{figures[label][2]:,.0f}")
        report.item(f"{label}  s/run, calls",
                    f"{statistics.median(r['seconds'] for r in runs):.1f}  "
                    f"{mean(r['calls'] for r in runs):.1f}")
        tokens = [mean(r["tokens"][i] for r in runs) for i in range(3)]
        report.item(f"{label}  tokens in / cached / out",
                    f"{tokens[0]:,.0f} / {tokens[1]:,.0f} / {tokens[2]:,.0f}")
        report.item(f"{label}  cache read share of input",
                    f"{tokens[1] / tokens[0]:.3f}" if tokens[0] else "-")
    report.item("all-generate over the agent, $",
                ratio(figures["all-generate"][0], figures["agent"][0]))
    report.item("all-generate over IR, $", ratio(figures["all-generate"][0], figures["IR"][0]))
    report.item("IR + chat over IR, $",
                ratio(figures["IR + chat decisions"][0], figures["IR"][0]))
    report.item("agent over IR + chat, $",
                ratio(figures["agent"][0], figures["IR + chat decisions"][0]))
    report.item("agent over all-generate, $",
                ratio(figures["agent"][0], figures["all-generate"][0]))
    for label, values in figures.items():
        report.item(f"{label}  paid / uncached (cache factor)", f"{values[3]:.1f}x")


def coverage(report: Report, plan: dict) -> None:
    report.section("coverage: held-out success against the number of traces")
    for family in plan["families"]:
        cells = []
        for k in K_VALUES:
            runs = [A.workflow_run(Path(p)) for p in sorted(
                glob.glob(str(COVERAGE / family / f"k{k}" / "runs" / "*")))]
            cells.append(f"k={k}: {sum(r['success'] for r in runs)}/{len(runs)}"
                         if runs else f"k={k}: -")
        report.item(f"{family}", "   ".join(cells))
    for family in plan["families"]:
        runs = [A.workflow_run(Path(p)) for p in sorted(
            glob.glob(str(COVERAGE / family / "k1" / "runs" / "*")))]
        if runs and sum(r["success"] for r in runs) < len(runs):
            report.item(f"{family} at k=1, tests passed/total",
                        f"{sum(r['tests_passed'] for r in runs)}/"
                        f"{sum(r['tests_total'] for r in runs)}")


def stability(report: Report, plan: dict) -> None:
    report.section("stability: the same traces, compiled again")
    packages = 0
    for family in plan["families"]:
        cells = []
        for run in sorted(glob.glob(str(STABILITY / family / "run*"))):
            runs = [A.workflow_run(Path(p)) for p in sorted(glob.glob(f"{run}/runs/*"))]
            packages += 1
            cells.append(f"{sum(r['success'] for r in runs)}/{len(runs)}"
                         if runs else "-")
        report.item(f"{family}  recompile 1 / 2", "  ".join(cells))
    report.item("recompiles", packages)


def graphs(report: Report, plan: dict) -> None:
    """The shape of each package, and the GENERATE nodes the coverage compiles use."""
    report.section("graphs: node counts, and GENERATE nodes in the coverage compiles")
    for family in plan["families"]:
        cells = [str(len(package(family)["nodes"]))]
        for run in sorted(glob.glob(str(STABILITY / family / "run*"))):
            path = Path(run) / "compile/workspace/out/graph.json"
            if path.exists():
                cells.append(str(len(json.loads(path.read_text())["nodes"])))
        report.item(f"{family}  nodes, main / recompile 1 / 2", " / ".join(cells))
    packages = 0
    runs = 0
    successes = 0
    for family in plan["families"]:
        for k in K_VALUES:
            path = COVERAGE / family / f"k{k}" / "compile/workspace/out/graph.json"
            if not path.exists():
                continue
            graph = json.loads(path.read_text())
            ids = [n["id"] for n in graph["nodes"] if n["type"] == "GENERATE"]
            if not ids:
                continue
            records = [A.workflow_run(Path(p)) for p in sorted(
                glob.glob(str(COVERAGE / family / f"k{k}" / "runs" / "*")))]
            packages += 1
            runs += len(records)
            successes += sum(r["success"] for r in records)
            report.item(f"{family} k={k} GENERATE",
                        f"{','.join(ids)}  {sum(r['success'] for r in records)}/{len(records)}")
    report.item("coverage packages with a GENERATE node", packages)
    report.item("held-out runs of those packages", runs)
    report.item("successes in those runs", successes)
    graph = json.loads((COVERAGE / "ce359b5" / "k3" / "compile/workspace/out/graph.json").read_text())
    counts = {kind: sum(1 for n in graph["nodes"] if n["type"] == kind)
              for kind in ("EXECUTE", "DECIDE", "GENERATE")}
    report.item("Songs k=3 package",
                f"{len(graph['nodes'])} nodes: {counts['EXECUTE']} EXECUTE, "
                f"{counts['DECIDE']} DECIDE, {counts['GENERATE']} GENERATE")


def one_session(path: str) -> float:
    total = 0.0
    for line in Path(path).read_text().splitlines():
        record = json.loads(line) if line.strip() else {}
        message = record.get("message") if record.get("type") == "message" else None
        if not message or message.get("role") != "assistant":
            continue
        usage = message.get("usage", {})
        total += cost("deepseek-flash", input_tokens=usage.get("input", 0),
                      output_tokens=usage.get("output", 0),
                      cache_hit_tokens=usage.get("cacheRead", 0))
    return total


def compile_spend() -> tuple[float, int]:
    """What every compile cost, wherever its session was recorded.

    Compiles live in three places: the main results, the coverage sweep and the
    stability sweep, each with its own records directory.
    """
    total, number = 0.0, 0
    patterns = (
        "compile/*/*/*/run/session/*.jsonl",
        "coverage/*/*/k*/compile/run/session/*.jsonl",
        "stability/*/*/run*/compile/run/session/*.jsonl",
    )
    for pattern in patterns:
        for path in glob.glob(str(ROOT / "runs/appworld" / pattern)):
            number += 1
            total += one_session(path)
    return total, number


def spend(report: Report, plan: dict, experiment: str) -> None:
    report.section("what the experiment cost (as paid)")
    cash, number = compile_spend()
    report.item("compiles, $ / count", f"{cash:.3f} / {number}")
    trace_cash = sum(A.agent_cost(run["meta"]) for family in plan["families"]
                     for run in trace_runs(family))
    report.item("traces, $", f"{trace_cash:.3f}")

    def total(store, reader, subdir="", experiment=None):
        runs = [run for family in plan["families"]
                for run in A.family_runs(store, family, reader, experiment=experiment,
                                         subdir=subdir)]
        return sum(run["cost"] for run in runs), len(runs)

    agent_cash, _ = total(A.AGENT, A.agent_run)
    report.item("agent on the held-out set, $", f"{agent_cash:.3f}")
    generate_cash, _ = total(A.ALL_GENERATE, A.generate_run)
    chat_cash, _ = total(A.CHAT_DECIDE, A.workflow_run)
    report.item("all-generate + IR + chat, $", f"{generate_cash + chat_cash:.3f}")
    ir_cash, _ = total(A.RUNS, A.workflow_run, "ir", experiment)
    report.item("IR on the held-out set, $", f"{ir_cash:.4f}")
    coverage_cash = sum(A.workflow_run(Path(path))["cost"] for path in
                        glob.glob(str(COVERAGE / "*" / "k*" / "runs" / "*")))
    stability_cash = sum(A.workflow_run(Path(path))["cost"] for path in
                         glob.glob(str(STABILITY / "*" / "run*" / "runs" / "*")))
    report.item("coverage + stability runs, $", f"{coverage_cash + stability_cash:.4f}")
    report.item("total, $", f"{cash + trace_cash + agent_cash + generate_cash + chat_cash + ir_cash + coverage_cash + stability_cash:.2f}")


def derived(report: Report, plan: dict) -> None:
    report.section("numbers derived, not recorded")
    ten = len(next(iter(plan["families"].values()))["heldout"])
    report.item(f"95% one-sided bound on success given {ten}/{ten}",
                f"{0.05 ** (1 / ten):.2f}")
    report.item("95% one-sided bound on success given 30/30", f"{0.05 ** (1 / 30):.2f}")
    per_compile = []
    for path in glob.glob(str(ROOT / "runs/appworld/compile/*/*/*/run/session/*.jsonl")):
        per_compile.append(one_session(path))
    report.item("one main compile, $ min / median / max",
                f"{min(per_compile):.3f} / {statistics.median(per_compile):.3f} / "
                f"{max(per_compile):.3f}")
    report.item("prices read on", READ_ON)
    report.item("price fingerprint", fingerprint()[:12])


def build(plan: dict, experiment: str) -> Report:
    report = Report()
    report.item("records", "runs/appworld", 46)
    instances(report, plan)
    traces(report, plan)
    compiles(report, plan)
    main_runs(report, plan, experiment)
    ladder(report, plan, experiment)
    coverage(report, plan)
    stability(report, plan)
    graphs(report, plan)
    spend(report, plan, experiment)
    derived(report, plan)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", default="p1")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="accept the current numbers")
    mode.add_argument("--print", dest="only_print", action="store_true",
                      help="print without comparing")
    arguments = parser.parse_args()

    plan = json.loads((A.PLANS / f"{arguments.experiment}.json").read_text())
    report = build(plan, arguments.experiment)
    printed = report.text()

    if arguments.write:
        AUDIT.write_text(printed)
        print(f"wrote {AUDIT.relative_to(ROOT)}")
    elif arguments.only_print:
        print(printed, end="")
    else:
        if not AUDIT.exists():
            print(f"{AUDIT.relative_to(ROOT)} is missing; run with --write", file=sys.stderr)
            raise SystemExit(1)
        if printed != AUDIT.read_text():
            expected, actual = AUDIT.read_text().splitlines(), printed.splitlines()
            for number, (was, now) in enumerate(zip(expected, actual)):
                if was != now:
                    print(f"line {number + 1}:\n  paper:  {was}\n  records: {now}")
            if len(expected) != len(actual):
                print(f"the report has {len(actual)} lines, the file {len(expected)}")
            raise SystemExit(1)
        print(f"{AUDIT.relative_to(ROOT)} matches the records")

    if report.failures:
        print("failed checks: " + ", ".join(report.failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
