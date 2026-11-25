"""Figures, generated from the records rather than typed.

Three drawings, each answering a question the tables answer in words:

- the compiled graphs, so a reader can see what compilation produces;
- the per-run cost of every arm on a log axis, so the spread is visible and not
  just the mean;
- held-out success and run cost against the number of traces, which is the
  question a reader with a task of their own actually asks.

Nothing here reads a number from a file a person wrote. Everything comes from
`runs/`, so a re-run or a price change regenerates the figures.
"""

import argparse
import glob
import json
import statistics
import subprocess
from pathlib import Path

from bench.appworld import analyze as A

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
COMPILES = ROOT / "runs/appworld/compile"
COVERAGE = ROOT / "runs/appworld/coverage"

ARMS = (
    ("all-generate", "all-generate", A.ALL_GENERATE, A.generate_run),
    ("agent", "agent", A.AGENT, A.agent_run),
    ("IR + chat", "IR-chat", A.CHAT_DECIDE, A.workflow_run),
    ("IR", "IR", A.RUNS, A.workflow_run),
)


def newest_package(family: str) -> Path:
    return sorted(COMPILES.glob(f"*/{family}/*/workspace/out"))[-1]


def dot_for(graph: dict) -> str:
    """A Graphviz program for one compiled workflow.

    EXECUTE is a box, DECIDE a diamond, GENERATE a parallelogram; a route is
    labelled with the answer that takes it, so a reader can see whether the
    decision branches or merely supplies a value.
    """
    shapes = {"EXECUTE": "box", "DECIDE": "diamond", "GENERATE": "parallelogram"}
    fills = {"EXECUTE": "#eef2f7", "DECIDE": "#fde9d0", "GENERATE": "#e6f2e6"}
    lines = [
        "digraph workflow {",
        '  rankdir=TB; nodesep=0.18; ranksep=0.26;',
        '  node [fontname="Helvetica", fontsize=9, style="filled", penwidth=0.7];',
        '  edge [fontname="Helvetica", fontsize=7, penwidth=0.7, arrowsize=0.6];',
    ]
    for node in graph["nodes"]:
        kind = node["type"]
        label = node["id"].replace("_", "\\n")
        lines.append(
            f'  "{node["id"]}" [shape={shapes.get(kind, "box")}, '
            f'fillcolor="{fills.get(kind, "#ffffff")}", label="{label}"];'
        )
    for node in graph["nodes"]:
        if isinstance(node.get("next"), str):
            lines.append(f'  "{node["id"]}" -> "{node["next"]}";')
        for answer, target in (node.get("routes") or {}).items():
            if isinstance(target, str):
                lines.append(f'  "{node["id"]}" -> "{target}" [label="{answer}"];')
    lines.append("}")
    return "\n".join(lines)


def successors(graph: dict) -> dict[str, list[tuple[str, str | None]]]:
    """Every outgoing edge of every node, with the answer that takes it."""
    edges: dict[str, list[tuple[str, str | None]]] = {n["id"]: [] for n in graph["nodes"]}
    for node in graph["nodes"]:
        if isinstance(node.get("next"), str):
            edges[node["id"]].append((node["next"], None))
        for answer, target in (node.get("routes") or {}).items():
            if isinstance(target, str):
                edges[node["id"]].append((target, answer))
    return edges


def depths(graph: dict, edges: dict) -> dict[str, int]:
    """Distance from the entrypoint, as a level; a cycle adds no second level."""
    depth = {graph["entrypoint"]: 0}
    queue = [graph["entrypoint"]]
    while queue:
        node = queue.pop(0)
        for target, _ in edges.get(node, []):
            if target not in depth:
                depth[target] = depth[node] + 1
                queue.append(target)
    for node in edges:
        depth.setdefault(node, 0)
    return depth


def best_cut(graph: dict, edges: dict, depth: dict[str, int]) -> int:
    """The level to break at: fewest edges crossing it, then the most even split."""
    levels = sorted(set(depth.values()))
    best = None
    for cut in levels[:-1]:
        crossing = sum(
            1
            for node, targets in edges.items()
            if depth[node] <= cut
            for target, _ in targets
            if depth[target] > cut
        )
        first = sum(1 for value in depth.values() if value <= cut)
        balance = abs(first - (len(depth) - first))
        key = (crossing, balance)
        if best is None or key < best[0]:
            best = (key, cut)
    return best[1] if best else levels[0]


def render_parts(graph: dict, name: str, parts: int = 2) -> int:
    """Render a long chain in rows, cutting between nodes so it stays legible."""
    shapes = {"EXECUTE": "box", "DECIDE": "diamond", "GENERATE": "parallelogram"}
    fills = {"EXECUTE": "#eef2f7", "DECIDE": "#fde9d0", "GENERATE": "#e6f2e6"}
    edges = successors(graph)
    depth = depths(graph, edges)
    cut = best_cut(graph, edges, depth)
    types = {n["id"]: n["type"] for n in graph["nodes"]}
    order = sorted(types, key=lambda node: (depth[node], node))
    chunks = [
        [node for node in order if depth[node] <= cut],
        [node for node in order if depth[node] > cut],
    ]
    nodes = order
    for index, chunk in enumerate(chunks, start=1):
        lines = [
            "digraph part {",
            '  rankdir=LR; nodesep=0.12; ranksep=0.22;',
            '  node [fontname="Helvetica", fontsize=12, style="filled", penwidth=0.8];',
            '  edge [fontname="Helvetica", fontsize=10, penwidth=0.8, arrowsize=0.7];',
        ]
        if index > 1:
            lines.append('  "_from" [shape=plaintext, style="", label="\u2026", fontsize=16];')
        for node in chunk:
            kind = types[node]
            lines.append(
                f'  "{node}" [shape={shapes.get(kind, "box")}, '
                f'fillcolor="{fills.get(kind, "#ffffff")}", label="{node.replace("_", chr(92)+"n")}"];'
            )
        if index > 1:
            lines.append(f'  "_from" -> "{chunk[0]}";')
        for node in chunk:
            for target, answer in edges.get(node, []):
                if target in chunk:
                    label = f' [label="{answer}"]' if answer else ""
                    lines.append(f'  "{node}" -> "{target}"{label};')
                elif target in nodes and nodes.index(target) > nodes.index(node):
                    lines.append(f'  "{node}" -> "_cont{index}";')
        if index < len(chunks):
            lines.append(f'  "_cont{index}" [shape=plaintext, style="", label="\u2026", fontsize=16];')
        lines.append("}")
        source = OUT / f"{name}-{index}.dot"
        source.write_text("\n".join(lines) + "\n")
        subprocess.run(["dot", "-Tpdf", str(source), "-o", str(OUT / f"{name}-{index}.pdf")], check=True)
    return len(nodes)


def render_graphs() -> None:
    """Two graphs: one that keeps decisions, one that has none."""
    for family, name in (("07b42fd", "graph-artists"), ("302c169", "graph-alarm"), ("ce359b5", "graph-songs")):
        graph = json.loads((newest_package(family) / "graph.json").read_text())
        count = render_parts(graph, name)
        print(f"{name}: {count} nodes, two rows")


def render_generate_graph() -> None:
    """The song family from three traces: the parse of the year is a GENERATE node."""
    package = COVERAGE / "pi-dsv41flash-high-v1" / "ce359b5" / "k3" / "compile" / "workspace" / "out"
    graph = json.loads((package / "graph.json").read_text())
    count = render_parts(graph, "graph-songs-k3")
    print(f"graph-songs-k3: {count} nodes, two rows")


def break_even() -> None:
    """Accumulated cost against executions: the agent line and the compiled line."""
    for family, name in (("07b42fd", "artists"), ("302c169", "alarms"), ("ce359b5", "songs")):
        ir = A.family_runs(A.RUNS, family, A.workflow_run, experiment="p1", subdir="ir")
        agent = A.family_runs(A.AGENT, family, A.agent_run)
        compile_cost, _ = A.compile_cost(family)
        invested = compile_cost + A.trace_cost(family)
        a = statistics.mean(run["cost"] for run in agent)
        r = statistics.mean(run["cost"] for run in ir)
        lines = [f"{x}\t{a * x:.8f}\t{invested + r * x:.8f}" for x in range(0, 61)]
        (OUT / f"break-even-{name}.dat").write_text("\n".join(lines) + "\n")
        print(f"break-even-{name}: invested {invested:.4f}, agent {a:.6f}, ir {r:.6f}")


def composition() -> None:
    """The nodes of nine packages, split by category."""
    rows = []
    for family in ("07b42fd", "302c169", "ce359b5"):
        paths = [newest_package(family) / "graph.json"]
        for run in sorted(glob.glob(str(ROOT / "runs/appworld/stability/pi-dsv41flash-high-v1" / family / "run*"))):
            paths.append(Path(run) / "compile/workspace/out/graph.json")
        for index, path in enumerate(paths, start=1):
            graph = json.loads(path.read_text())
            counts = {kind: sum(1 for n in graph["nodes"] if n["type"] == kind)
                      for kind in ("EXECUTE", "DECIDE", "GENERATE")}
            rows.append(f"{len(rows) + 1}\t{counts['EXECUTE']}\t{counts['DECIDE']}\t{counts['GENERATE']}")
    (OUT / "composition.dat").write_text("\n".join(rows) + "\n")
    print(f"composition: {len(rows)} packages")


def cost_strip() -> None:
    """Every instance's cost, per arm, with the median marked."""
    instances, medians = [], []
    for label, _, store, read in ARMS:
        runs = []
        for family in ("07b42fd", "302c169", "ce359b5"):
            if store is A.RUNS:
                runs += A.family_runs(store, family, read, experiment="p1", subdir="ir")
            else:
                runs += A.family_runs(store, family, read)
        index = len(medians) + 1
        for run in runs:
            # A run that makes no model call costs exactly nothing, which a
            # logarithmic axis cannot show; the median is computed over all runs.
            if run["cost"] > 0:
                instances.append(f"{index}\t{run['cost']:.8f}")
        medians.append(f"{index}\t{statistics.median(r['cost'] for r in runs):.8f}")
    (OUT / "cost-instances.dat").write_text("\n".join(instances) + "\n")
    (OUT / "cost-medians.dat").write_text("\n".join(medians) + "\n")
    print(f"cost strip: {len(instances)} points")


def coverage(plan: dict) -> None:
    """Held-out success and run cost against the number of traces.

    One file per family, because pgfplots reads one table per plot; the k column
    becomes an index so the axis can label it with the values actually used.
    """
    index = {"1": 1, "3": 2, "5": 3, "10": 4, "all": 5}
    points = 0
    for family in plan["families"]:
        lines = []
        for k, position in index.items():
            base = COVERAGE / "pi-dsv41flash-high-v1" / family / f"k{k}"
            runs = [A.workflow_run(Path(p)) for p in sorted(glob.glob(str(base / "runs" / "*")))]
            if not runs:
                continue
            solved = sum(run["success"] for run in runs)
            mean = sum(run["cost"] for run in runs) / len(runs)
            lines.append(f"{position}\t{solved}\t{mean:.8f}")
        (OUT / f"coverage-{family}.dat").write_text("\n".join(lines) + "\n")
        points += len(lines)
    print(f"coverage: {points} points")


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", default="p1")
    parser.add_argument("--out", type=Path, default=None, help="default: figures/")
    arguments = parser.parse_args()
    OUT = arguments.out or OUT
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / "plans" / f"{arguments.plan}.json").read_text())
    render_graphs()
    render_generate_graph()
    break_even()
    composition()
    cost_strip()
    coverage(plan)


if __name__ == "__main__":
    main()
