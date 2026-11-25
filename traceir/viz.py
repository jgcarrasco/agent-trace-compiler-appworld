"""Draw a workflow: Mermaid or Graphviz DOT text, and images via graphviz.

Text rendering is pure: no files, no subprocess, no model. Only the CLI shells
out to `dot`, and only for --format svg|png. A run can be overlaid by passing
visit counts, which keeps this module free of engine types.
"""

import argparse
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from traceir.ir.models import (
    ChoiceQuestion,
    DecideNode,
    ExecuteNode,
    GenerateNode,
    Reference,
    ScoreQuestion,
    Workflow,
)
from traceir.ir.validate import load_graph, validate_graph

SHAPES = {"EXECUTE": "box", "GENERATE": "ellipse", "DECIDE": "diamond"}


def _summary(node: ExecuteNode | DecideNode | GenerateNode) -> str:
    """The one line that says what the node is."""
    if isinstance(node, GenerateNode):
        return "GENERATE"
    if isinstance(node, DecideNode):
        question = node.question
        if isinstance(question, ChoiceQuestion):
            criteria = question.criteria
            if isinstance(criteria, Reference):
                return f"choice from {criteria.ref}"
            if "routes" in node.model_fields_set:
                return "choice"  # the outgoing edges already carry these names
            return f"choice: {', '.join(criteria)}"
        if isinstance(question, ScoreQuestion):
            criteria = question.criteria
            if isinstance(criteria, Reference):
                return f"score from {criteria.ref}"
            return f"score 0..{len(criteria) - 1}"
        criteria = question.criteria
        if isinstance(criteria, Reference):
            return f"noul from {criteria.ref}"
        return "noul 0..1"
    return "EXECUTE" if node.operation == node.id else f"EXECUTE {node.operation}"


def _edges(workflow: Workflow) -> Iterator[tuple[str, str | None, str | None]]:
    """(source, label, target) triples. A None target ends the workflow."""
    for node in workflow.nodes:
        if "routes" in node.model_fields_set:
            for name, target in node.routes.items():
                yield node.id, name, target
        elif isinstance(node, DecideNode) and "bands" in node.model_fields_set:
            for band in node.bands:
                label = "else" if band.max is None else f"<= {band.max:g}"
                yield node.id, label, band.next
        else:
            yield node.id, None, node.next


def _back_edges(workflow: Workflow) -> set[tuple[str, str | None, str | None]]:
    """The edges that close a cycle, found by walking the graph from the entrypoint.

    Marking one `constraint=false` keeps it out of the rank ordering and, more
    usefully, stops graphviz drawing it on top of the forward edge between the
    same pair of nodes.
    """
    by_source: dict[str, list[tuple[str, str | None, str | None]]] = {}
    for edge in _edges(workflow):
        by_source.setdefault(edge[0], []).append(edge)
    closing: set[tuple[str, str | None, str | None]] = set()
    on_path: set[str] = set()
    seen: set[str] = set()

    def walk(node_id: str) -> None:
        on_path.add(node_id)
        seen.add(node_id)
        for edge in by_source.get(node_id, []):
            target = edge[2]
            if target is None:
                continue
            if target in on_path:
                closing.add(edge)
            elif target not in seen:
                walk(target)
        on_path.discard(node_id)

    walk(workflow.entrypoint)
    return closing


def _unique(preferred: str, taken: set[str]) -> str:
    """A name that does not collide with what is already in use."""
    candidate, suffix = preferred, 2
    while candidate in taken:
        candidate, suffix = f"{preferred}_{suffix}", suffix + 1
    return candidate


def _labels(workflow: Workflow) -> tuple[dict[str, str], str, str]:
    """Mermaid-safe identifiers, plus the ones for the input and output nodes.

    Mermaid reserves `end` and dislikes punctuation in ids, so every id is
    prefixed and sanitized; the real id is in the visible label.
    """
    ids: dict[str, str] = {}
    for node in workflow.nodes:
        preferred = "n_" + "".join(c if c.isalnum() else "_" for c in node.id)
        ids[node.id] = _unique(preferred, set(ids.values()))
    used = set(ids.values())
    return ids, _unique("n_input", used), _unique("n_output", used)


def mermaid(workflow: Workflow, visits: Counter[str] | None = None) -> str:
    counts = visits or Counter()
    ids, source_id, sink_id = _labels(workflow)
    shapes = {
        "EXECUTE": lambda name, label: f'{name}["{label}"]',
        "GENERATE": lambda name, label: f'{name}(["{label}"])',
        "DECIDE": lambda name, label: f'{name}{{"{label}"}}',
    }
    lines = ["flowchart TD"]
    lines += [
        f"  {shapes[node.type](ids[node.id], _mermaid_text(_caption(node), counts[node.id]))}"
        for node in workflow.nodes
    ]
    lines.append(f'  {source_id}[/"{_mermaid_text(_field_caption("input", workflow.input_schema), 0)}"/]')
    lines.append(f'  {sink_id}[/"{_mermaid_text(_field_caption("output", workflow.output_schema), 0)}"/]')
    lines.append(f"  {source_id} -.-> {ids[workflow.entrypoint]}")
    for source, label, target in _edges(workflow):
        arrow = f" -->|{_mermaid_text(label, 0)}| " if label else " --> "
        lines.append(f"  {ids[source]}{arrow}{sink_id if target is None else ids[target]}")
    return "\n".join(lines) + "\n"


def dot(workflow: Workflow, visits: Counter[str] | None = None) -> str:
    counts = visits or Counter()
    taken = {node.id for node in workflow.nodes}
    closing = _back_edges(workflow)
    source_id, sink_id = _unique("__input__", taken), _unique("__output__", taken)
    lines = [
        "digraph workflow {",
        "  rankdir=TB;",
        '  node [shape=box, fontname="Helvetica", fontsize=10, margin="0.12,0.06"];',
        '  edge [fontname="Helvetica", fontsize=9];',
        f'  {_quote(source_id)} [shape=parallelogram, style=filled, fillcolor="#eef4fb", '
        f'label={_quote(_field_caption("input", workflow.input_schema))}];',
        f'  {_quote(sink_id)} [shape=parallelogram, style=filled, fillcolor="#eef4fb", '
        f'label={_quote(_field_caption("output", workflow.output_schema))}];',
        f'  {_quote(source_id)} -> {_quote(workflow.entrypoint)} [style=dashed];',
    ]
    for node in workflow.nodes:
        extra = f"\nx{counts[node.id]}" if counts[node.id] > 1 else ""
        lines.append(f"  {_quote(node.id)} [shape={SHAPES[node.type]}, label={_quote(_caption(node) + extra)}];")
    for source, label, target in _edges(workflow):
        to = sink_id if target is None else target
        attributes = [] if label is None else [f"label={_quote(label)}"]
        if (source, label, target) in closing:
            attributes.append("constraint=false")
        arrow = f" -> {_quote(to)}"
        if attributes:
            arrow += " [" + ", ".join(attributes) + "]"
        lines.append(f"  {_quote(source)}{arrow};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _fields(schema: object) -> str:
    """Declared top-level fields of an input or output schema, required ones starred."""
    if not isinstance(schema, dict):
        return ""  # a boolean schema states no fields
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return ""
    required = schema.get("required")
    required = set(required) if isinstance(required, list) else set()
    names = [f"{name}*" if name in required else str(name) for name in properties]
    shown, hidden = names[:6], len(names) - 6
    return ", ".join(shown + ([f"+{hidden} more"] if hidden > 0 else []))


def _caption(node: ExecuteNode | DecideNode | GenerateNode) -> str:
    """Both lines of a node: its id, then what kind of node it is."""
    return f"{node.id}\n{_summary(node)}"


def _field_caption(kind: str, schema: object) -> str:
    fields = _fields(schema)
    return f"{kind}\n{fields}" if fields else kind


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _mermaid_text(value: str, count: int) -> str:
    label = "<br/>".join(line.replace('"', "'").replace("|", "/") for line in value.split("\n"))
    return f"{label}<br/>x{count}" if count > 1 else label


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog="python -m traceir.viz")
    parser.add_argument("graph", type=Path)
    parser.add_argument("--format", choices=("mermaid", "dot", "svg", "png"), default="mermaid")
    parser.add_argument("--out", type=Path, default=None, help="Default: stdout, or <graph>.<format>")
    arguments = parser.parse_args(argv)

    try:
        workflow = validate_graph(load_graph(arguments.graph))
    except (OSError, ValueError) as error:
        print(f"Invalid IR: {error}", file=sys.stderr)
        return 1

    if arguments.format in ("mermaid", "dot"):
        text = mermaid(workflow) if arguments.format == "mermaid" else dot(workflow)
        if arguments.out is None:
            print(text, end="")
        else:
            arguments.out.write_text(text)
            print(arguments.out)
        return 0

    if shutil.which("dot") is None:
        print("graphviz `dot` not found; install graphviz to render svg/png", file=sys.stderr)
        return 1
    rendered = subprocess.run(
        ["dot", f"-T{arguments.format}"], input=dot(workflow).encode(), capture_output=True
    )
    if rendered.returncode != 0:
        print(f"dot failed: {rendered.stderr.decode(errors='replace').strip()}", file=sys.stderr)
        return 1
    output = arguments.out or arguments.graph.with_suffix(f".{arguments.format}")
    output.write_bytes(rendered.stdout)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
