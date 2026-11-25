"""Rendering is string building, so these tests compare text and never call dot."""

import io
import json
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from traceir.ir.models import Workflow
from traceir.viz import _back_edges, _edges, dot, main, mermaid
from traceir.ir.validate import load_graph, validate_graph

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "workflows" / "email_reply" / "graph.json"


def build(nodes: list[dict], entrypoint: str | None = None) -> Workflow:
    return validate_graph(
        {
            "schema_version": "0.1",
            "entrypoint": entrypoint or nodes[0]["id"],
            "input_schema": {"type": "object"},
            "output_schema": {"type": "string"},
            "nodes": nodes,
        },
        operations={"noop"},
    )


def execute(node_id: str, **extra) -> dict:
    return {
        "type": "EXECUTE",
        "id": node_id,
        "inputs": {},
        "operation": "noop",
        **extra,
    }


def decide(node_id: str, question: dict, **extra) -> dict:
    return {"type": "DECIDE", "id": node_id, "inputs": {}, "question": question, **extra}


CHOICE = {
    "type": "choice",
    "instructions": "Pick one.",
    "criteria": {"A": "first", "B": "second"},
}


class ExampleGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = validate_graph(load_graph(EXAMPLE))

    def test_mermaid_shows_every_node_with_its_type(self):
        text = mermaid(self.workflow)
        self.assertTrue(text.startswith("flowchart TD\n"))
        self.assertIn('n_draft_reply(["draft_reply<br/>GENERATE"])', text)
        self.assertIn('n_prepare_review["prepare_review<br/>EXECUTE"]', text)
        self.assertIn('n_ask_approval["ask_approval<br/>EXECUTE request_input"]', text)
        self.assertIn('n_interpret_approval{"interpret_approval<br/>choice"}', text)

    def test_mermaid_labels_branches_and_ends_the_workflow(self):
        text = mermaid(self.workflow)
        self.assertIn("n_interpret_approval -->|APPROVE| n_send_reply", text)
        self.assertIn("n_interpret_approval -->|UNCLEAR| n_ask_approval", text)
        self.assertIn("n_send_reply --> n_output", text)
        self.assertIn('n_input[/"input<br/>email*, instruction*"/]', text)
        self.assertIn('n_output[/"output<br/>status*, message_id*"/]', text)
        self.assertIn("n_input -.-> n_draft_reply", text)  # dashed: not a node of the graph

    def test_dot_declares_shapes_and_edge_labels(self):
        text = dot(self.workflow)
        self.assertTrue(text.startswith("digraph workflow {\n"))
        self.assertIn("rankdir=TB;", text)
        self.assertIn('"draft_reply" [shape=ellipse, label="draft_reply\nGENERATE"];', text)
        self.assertIn('"interpret_approval" [shape=diamond', text)
        self.assertIn('"interpret_approval" -> "send_reply" [label="APPROVE"];', text)
        self.assertIn('"send_reply" -> "__output__";', text)
        self.assertIn('[shape=parallelogram, style=filled', text)
        self.assertIn('"__input__" -> "draft_reply" [style=dashed];', text)

    def test_visits_are_annotated_and_absent_by_default(self):
        self.assertNotIn("x2", mermaid(self.workflow))
        self.assertIn("ask_approval<br/>EXECUTE request_input<br/>x2", mermaid(
            self.workflow, Counter({"ask_approval": 2})))
        self.assertIn('label="ask_approval\nEXECUTE request_input\nx2"',
                      dot(self.workflow, Counter({"ask_approval": 2})))


    def test_a_back_edge_is_let_out_of_the_rank_ordering(self):
        text = dot(self.workflow)
        self.assertIn('"interpret_approval" -> "ask_approval" [label="UNCLEAR", constraint=false];', text)
        self.assertIn('"ask_approval" -> "interpret_approval";', text)  # the forward twin stays plain


class CommittedDiagramTests(unittest.TestCase):
    """The example's rendered diagram is checked in, so check that it is current.

    Comparing SVG bytes would tie the suite to a graphviz version. The node and
    edge titles graphviz embeds are the structure, and that is what should not
    drift from graph.json.
    """

    diagram = ROOT / "workflows" / "email_reply" / "graph.svg"

    def test_diagram_matches_the_graph_it_was_drawn_from(self):
        workflow = validate_graph(load_graph(EXAMPLE))
        root = ET.fromstring(self.diagram.read_text())
        # graphviz names every node and edge, inside the SVG namespace.
        titles = {element.text for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "title"}
        self.assertTrue(titles, "no titles parsed: is this still a graphviz SVG?")
        drawn_nodes = (
            {node.id for node in workflow.nodes} | {"workflow", "__input__", "__output__"}
        )
        drawn_edges = {
            f"{source}->{target if target is not None else '__output__'}"
            for source, _, target in _edges(workflow)
        } | {f"__input__->{workflow.entrypoint}"}
        expected = drawn_nodes | drawn_edges
        self.assertEqual(titles - expected, set(), "the diagram shows something the graph does not have")
        self.assertEqual(
            expected - titles,
            set(),
            "regenerate it: python -m traceir.viz workflows/email_reply/graph.json --format svg",
        )


class InputAndOutputLabelsTests(unittest.TestCase):
    """The two pseudo-nodes report the declared contract, nothing invented."""

    def nodes(self):
        return [execute("only", next=None)]

    def build_with(self, input_schema, output_schema):
        document = {
            "schema_version": "0.1",
            "entrypoint": "only",
            "input_schema": input_schema,
            "output_schema": output_schema,
            "nodes": self.nodes(),
        }
        return validate_graph(document, operations={"noop"})

    def test_required_fields_are_starred(self):
        workflow = self.build_with(
            {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
             "required": ["a"]},
            {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        )
        text = dot(workflow)
        self.assertIn('label="input\na*, b"', text)
        self.assertIn('label="output\nok*"', text)

    def test_a_boolean_schema_states_no_fields(self):
        text = dot(self.build_with(True, False))
        self.assertIn('label="input"', text)
        self.assertIn('label="output"', text)

    def test_long_field_lists_are_capped(self):
        many = {f"f{i}": {"type": "string"} for i in range(9)}
        text = dot(self.build_with({"type": "object", "properties": many}, True))
        self.assertIn("f0, f1, f2, f3, f4, f5, +3 more", text)


class DecisionLabelTests(unittest.TestCase):
    """A node shows what its edges do not: the options only when nothing else does."""

    def text(self, **decision) -> str:
        return dot(build([decide("pick", CHOICE, write_to="/state/a", **decision),
                          execute("one", next=None), execute("two", next=None)]))

    def test_routed_choice_lets_the_edge_labels_speak(self):
        text = self.text(routes={"A": "one", "B": "two"})
        self.assertIn('label="pick\nchoice"', text)
        self.assertIn('[label="A"]', text)
        self.assertNotIn("A, B", text)

    def test_unbranched_choice_lists_its_options(self):
        text = self.text(next="one")
        self.assertIn('label="pick\nchoice: A, B"', text)

    def test_confidence_bands_keep_the_options(self):
        text = self.text(on="confidence", bands=[{"max": 0.5, "next": "one"}, {"max": None, "next": "two"}])
        self.assertIn('label="pick\nchoice: A, B"', text)  # edges say <= 0.5 / else instead


class BandAndCycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = build([
            decide("confidence_gate", CHOICE, on="confidence", write_to="/state/answer", bands=[
                {"max": 0.75, "next": "ask_user"},
                {"max": None, "next": "continue"},
            ]),
            execute("ask_user", next="confidence_gate"),
            execute("continue", next=None),
        ])

    def test_band_maxima_become_edge_labels(self):
        text = mermaid(self.workflow)
        self.assertIn("n_confidence_gate -->|<= 0.75| n_ask_user", text)
        self.assertIn("n_confidence_gate -->|else| n_continue", text)
        self.assertIn('"confidence_gate" -> "ask_user" [label="<= 0.75"];', dot(self.workflow))
        self.assertIn('"confidence_gate" -> "continue" [label="else"];', dot(self.workflow))

    def test_back_edge_is_present(self):
        self.assertIn("n_ask_user --> n_confidence_gate", mermaid(self.workflow))

    def test_only_the_closing_edge_of_a_cycle_is_marked(self):
        self.assertEqual(_back_edges(self.workflow), {("ask_user", None, "confidence_gate")})

    def test_a_straight_line_graph_has_no_back_edges(self):
        self.assertEqual(_back_edges(build([execute("a", next="b"), execute("b", next=None)])), set())

    def test_score_and_noul_summaries(self):
        workflow = build([
            decide("s", {"type": "score", "instructions": "Rate.", "criteria": ["low", "mid", "high"]},
                   bands=[{"max": None, "next": None}]),
            decide("n", {"type": "noul", "instructions": "Likely?"}, bands=[{"max": None, "next": None}]),
            decide("r", {"type": "choice", "instructions": "Which?", "criteria": {"ref": "/state/c"}},
                   routes={"A": "s"}),
            execute("s2", next=None),
        ], entrypoint="r")
        text = mermaid(workflow)
        self.assertIn("score 0..2", text)
        self.assertIn("noul 0..1", text)
        self.assertIn("choice from /state/c", text)


class IdentifierTests(unittest.TestCase):
    def test_mermaid_ids_are_safe_and_unique(self):
        workflow = build([
            execute("a-b", next="a_b"),
            execute("a_b", next="output"),
            execute("output", next=None),
        ])
        text = mermaid(workflow)
        declared = [line.strip().split("[")[0] for line in text.splitlines() if "[" in line]
        self.assertEqual(len(declared), len(set(declared)), declared)
        self.assertIn('n_output["output<br/>EXECUTE noop"]', text)  # the real node keeps the plain name
        self.assertIn('n_output_2[/"output"/]', text)  # the pseudo-node was pushed aside
        for source, target in (("n_a_b", "n_a_b_2"), ("n_a_b_2", "n_output"), ("n_output", "n_output_2")):
            self.assertIn(f"{source} --> {target}", text)

    def test_dot_quotes_ids_and_route_names(self):
        workflow = build([execute('odd"name', routes={'say "hi"': None})])
        text = dot(workflow)
        self.assertIn('"odd\\"name"', text)
        self.assertIn('[label="say \\"hi\\""]', text)


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.out = io.StringIO()

    def run_cli(self, argv: list[str]) -> int:
        with redirect_stdout(self.out):
            return main(argv)

    def test_text_formats_go_to_stdout_by_default(self):
        self.assertEqual(self.run_cli([str(EXAMPLE)]), 0)
        self.assertIn("flowchart TD", self.out.getvalue())

    def test_out_writes_the_file_and_reports_the_path(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "graph.mmd"
            self.assertEqual(self.run_cli([str(EXAMPLE), "--format", "mermaid", "--out", str(target)]), 0)
            self.assertEqual(self.out.getvalue().strip(), str(target))
            self.assertIn("flowchart TD", target.read_text())

    def test_missing_dot_is_reported_not_crashed(self):
        errors = io.StringIO()
        with patch("shutil.which", return_value=None), redirect_stderr(errors):
            self.assertEqual(self.run_cli([str(EXAMPLE), "--format", "svg"]), 1)
        self.assertIn("graphviz", errors.getvalue())

    def test_rendered_bytes_are_written(self):
        completed = subprocess.CompletedProcess([], 0, stdout=b"\x89PNG bytes", stderr=b"")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "graph.png"
            with patch("subprocess.run", return_value=completed) as run:
                self.assertEqual(self.run_cli([str(EXAMPLE), "--format", "png", "--out", str(target)]), 0)
            self.assertEqual(run.call_args.args[0], ["dot", "-Tpng"])
            self.assertEqual(target.read_bytes(), b"\x89PNG bytes")

    def test_dot_failure_is_reported(self):
        completed = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"syntax error at line 3")
        errors = io.StringIO()
        with patch("subprocess.run", return_value=completed), redirect_stderr(errors):
            self.assertEqual(self.run_cli([str(EXAMPLE), "--format", "svg"]), 1)
        self.assertIn("syntax error at line 3", errors.getvalue())

    def test_invalid_graph_does_not_render(self):
        broken = json.loads(EXAMPLE.read_text())
        broken["nodes"][3]["routes"]["APPROVE"] = "nowhere"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text(json.dumps(broken))
            errors = io.StringIO()
            with redirect_stderr(errors):
                self.assertEqual(self.run_cli([str(path)]), 1)
            self.assertIn("Invalid IR", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
