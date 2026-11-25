"""Offline IR artifact tests; no model, operation, or benchmark execution."""

import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from traceir.ir.models import DecideNode, ExecuteNode, GenerateNode, Workflow, graph_schema
from traceir.ir.validate import IRValidationError, load_graph, validate_graph

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "workflows/email_reply/graph.json"
SCHEMA = ROOT / "traceir/ir/schema.json"
OPERATIONS = {
    "prepare_review", "request_input", "send_reply", "report_not_sent",
}


class IRValidationTests(unittest.TestCase):
    def setUp(self):
        self.graph = load_graph(EXAMPLE)

    def approval(self):
        return self.graph["nodes"][3]

    def assert_invalid(self, text=None):
        with self.assertRaises(IRValidationError) as caught:
            validate_graph(self.graph)
        if text:
            self.assertIn(text, str(caught.exception))
        return caught.exception

    def test_generated_schema_matches_committed_file(self):
        self.assertEqual(json.loads(SCHEMA.read_text()), graph_schema())
        Draft202012Validator.check_schema(graph_schema())

    def test_complete_example_validates_and_round_trips(self):
        workflow = validate_graph(self.graph, operations=OPERATIONS)
        self.assertEqual(len(workflow.nodes), 6)
        self.assertEqual({node.type for node in workflow.nodes}, {"EXECUTE", "DECIDE", "GENERATE"})
        decision = workflow.nodes[3]
        assert isinstance(decision, DecideNode)
        self.assertEqual(decision.routes["UNCLEAR"], "ask_approval")
        self.assertEqual(workflow.model_dump(exclude_unset=True), self.graph)
        self.assertIsNone(workflow.nodes[-1].next)

    def test_flow_nodes_inherit_closed_model_configuration(self):
        for model in (ExecuteNode, GenerateNode):
            with self.subTest(model=model.__name__):
                self.assertEqual(model.model_config.get("extra"), "forbid")
                self.assertIs(model.model_config.get("strict"), True)
                self.assertIs(model.model_config.get("allow_inf_nan"), False)

    def test_example_input_and_final_output_contracts(self):
        Draft202012Validator(self.graph["input_schema"]).validate({
            "email": {"id": "e1", "sender": "marta@example.com", "body": "Meet Tuesday?"},
            "instruction": "Draft a reply confirming the time.",
        })
        validator = Draft202012Validator(self.graph["output_schema"])
        for value in (
            {"status": "submitted", "message_id": "m1"},
            {"status": "not_sent", "message_id": None},
        ):
            validator.validate(value)
        for value in (
            {"status": "submitted", "message_id": None},
            {"status": "not_sent", "message_id": "m1"},
            {"status": "delivered", "message_id": "m1"},
        ):
            self.assertFalse(validator.is_valid(value))

    def test_duplicate_node_ids(self):
        self.graph["nodes"].append(copy.deepcopy(self.graph["nodes"][0]))
        self.assert_invalid("duplicate node ID")

    def test_unknown_entrypoint(self):
        self.graph["entrypoint"] = "absent"
        self.assert_invalid("/entrypoint: unknown node ID")

    def test_unknown_successor_on_next_or_route(self):
        for kind in (
            "generate_next",
            "execute_next",
            "execute_routes",
            "decide_routes",
            "decide_bands",
        ):
            with self.subTest(kind=kind):
                self.graph = load_graph(EXAMPLE)
                if kind == "generate_next":
                    self.graph["nodes"][0]["next"] = "absent"
                elif kind == "execute_next":
                    self.graph["nodes"][1]["next"] = "absent"
                elif kind == "execute_routes":
                    del self.graph["nodes"][1]["next"]
                    self.graph["nodes"][1]["routes"] = {"READY": "absent"}
                elif kind == "decide_routes":
                    self.approval()["routes"]["APPROVE"] = "absent"
                else:
                    node = self.approval()
                    del node["routes"]
                    node["question"] = {"type": "noul", "instructions": "Eligible?"}
                    node["bands"] = [
                        {"max": 0.4, "next": "absent"},
                        {"max": None, "next": "send_reply"},
                    ]
                self.assert_invalid("unknown successor")

    def test_invalid_node_type_and_version(self):
        self.graph["nodes"][0]["type"] = "AGENT"
        self.assert_invalid()
        self.graph = load_graph(EXAMPLE)
        self.graph["schema_version"] = "0.2"
        self.assert_invalid()

    def test_unknown_fields_are_rejected(self):
        for scope in ("graph", "node", "question"):
            with self.subTest(scope=scope):
                self.graph = load_graph(EXAMPLE)
                target = {"graph": self.graph, "node": self.graph["nodes"][0],
                          "question": self.approval()["question"]}[scope]
                target["unexpected"] = "value"
                self.assert_invalid()

    def test_exactly_one_continuation_is_required(self):
        for mode in ("both", "neither", "empty_routes"):
            with self.subTest(mode=mode):
                self.graph = load_graph(EXAMPLE)
                node = self.graph["nodes"][1]
                if mode == "both":
                    node["routes"] = {"READY": "ask_approval"}
                elif mode == "neither":
                    del node["next"]
                else:
                    del node["next"]
                    node["routes"] = {}
                self.assert_invalid()
                with self.assertRaises(ValidationError):
                    Workflow.model_validate(self.graph)

    def test_decide_declares_exactly_one_continuation(self):
        for mode in ("routes_and_bands", "next_and_routes", "next_and_bands", "none"):
            with self.subTest(mode=mode):
                self.graph = load_graph(EXAMPLE)
                node = self.approval()
                if mode == "routes_and_bands":
                    node["bands"] = [{"max": None, "next": "send_reply"}]
                elif mode == "next_and_routes":
                    node["next"] = "send_reply"
                elif mode == "next_and_bands":
                    del node["routes"]
                    node["next"] = "send_reply"
                    node["bands"] = [{"max": None, "next": "send_reply"}]
                else:
                    del node["routes"]
                self.assert_invalid("exactly one of")
                with self.assertRaises(ValidationError):
                    Workflow.model_validate(self.graph)

    def test_decide_still_rejects_generic_templates_and_model_names(self):
        for field, value in (("template", "Decide."), ("domain", {}), ("model", "jev-latest")):
            with self.subTest(field=field):
                self.graph = load_graph(EXAMPLE)
                self.approval()[field] = value
                self.assert_invalid()

    def test_reference_syntax_and_state_only_destinations(self):
        for reference in ("/environment/db", "state/draft", "/state/bad~2key", "/state/bad~"):
            with self.subTest(reference=reference):
                self.graph = load_graph(EXAMPLE)
                self.graph["nodes"][1]["inputs"]["body"] = {"ref": reference}
                self.assert_invalid()
        for destination in ("/input/instruction", "/state", "/state/bad~3key"):
            with self.subTest(destination=destination):
                self.graph = load_graph(EXAMPLE)
                self.graph["nodes"][0]["write_to"] = destination
                self.assert_invalid()

    def test_escaped_pointers_and_literal_values(self):
        self.graph["nodes"][1]["inputs"]["body"] = {"ref": "/state/a~1b/~0name"}
        self.graph["nodes"][1]["inputs"]["recipient"] = {"literal": {"ref": "ordinary data"}}
        validate_graph(self.graph)

    def test_binding_cannot_be_both_reference_and_literal(self):
        self.graph["nodes"][1]["inputs"]["body"] = {"ref": "/state/draft", "literal": "text"}
        self.assert_invalid()

    def test_operation_availability_is_optional_and_does_not_import_code(self):
        self.graph["nodes"][1]["operation"] = "nonexistent.module.function"
        validate_graph(self.graph)
        validate_graph(self.graph, operations=(OPERATIONS - {"prepare_review"}) | {"nonexistent.module.function"})
        with self.assertRaisesRegex(IRValidationError, "unavailable operation"):
            validate_graph(self.graph, operations=OPERATIONS)

    def test_interaction_operation_names_are_not_implicitly_available(self):
        available = OPERATIONS - {"request_input"}
        for operation in ("request_input", "ask_user", "custom.prompt"):
            with self.subTest(operation=operation):
                self.graph["nodes"][2]["operation"] = operation
                validate_graph(self.graph)
                with self.assertRaisesRegex(IRValidationError, "unavailable operation"):
                    validate_graph(self.graph, operations=available)
                validate_graph(self.graph, operations=available | {operation})

    def test_operation_argument_names_are_not_hardcoded(self):
        for operation in ("request_input", "ask_user", "custom.prompt"):
            for inputs in ({}, {"question": {"literal": "Send?"}},
                           {"message": {"literal": "Send?"}, "extra": {"literal": 1}}):
                with self.subTest(operation=operation, inputs=inputs):
                    self.graph["nodes"][2]["operation"] = operation
                    self.graph["nodes"][2]["inputs"] = inputs
                    validate_graph(self.graph, operations=OPERATIONS | {operation})

    def test_decide_carries_one_question_of_any_kind(self):
        for question in (
            {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "Payments", "technical": "Bugs"}},
            {"type": "score", "instructions": "How frustrated?",
             "criteria": ["Calm", "Frustrated", "Very angry"]},
            {"type": "noul", "instructions": "The request is time-sensitive",
             "criteria": {"true": "Explicit urgency", "false": "No urgency"}},
        ):
            with self.subTest(question_type=question["type"]):
                self.graph = load_graph(EXAMPLE)
                node = self.approval()
                node["question"] = question
                node["on"] = "answer"
                if question["type"] == "choice":
                    node["routes"] = {"billing": "send_reply", "technical": "report_not_sent"}
                else:
                    del node["routes"]
                    node["bands"] = [{"max": 0.5, "next": "send_reply"}, {"max": None, "next": None}]
                decision = validate_graph(self.graph).nodes[3]
                assert isinstance(decision, DecideNode)
                self.assertEqual(decision.question.type, question["type"])

    def test_noul_criteria_are_optional(self):
        self.approval()["question"] = {"type": "noul", "instructions": "Is it urgent?"}
        del self.approval()["routes"]
        self.approval()["bands"] = [{"max": 0.5, "next": "send_reply"}, {"max": None, "next": None}]
        validate_graph(self.graph)

    def test_dynamic_criteria_references_point_at_state(self):
        for question_type in ("choice", "score", "noul"):
            with self.subTest(question_type=question_type):
                self.graph = load_graph(EXAMPLE)
                node = self.approval()
                node["question"] = {
                    "type": question_type,
                    "instructions": "Evaluate the request",
                    "criteria": {"ref": "/state/runtime_criteria"},
                }
                if question_type == "choice":
                    # The labels are only known at run time, so the validator
                    # cannot compare them with the declared routes.
                    node["routes"] = {"WHATEVER": "send_reply"}
                else:
                    del node["routes"]
                    node["bands"] = [{"max": 0.5, "next": "send_reply"}, {"max": None, "next": None}]
                # Only syntax/declarations are checked; runtime values are not present.
                validate_graph(self.graph)
        # The question itself is decided at compile time, so it cannot be loaded
        # from state: only its criteria may be.
        self.approval()["question"] = {"ref": "/state/runtime_question"}
        self.assert_invalid()

    def test_invalid_question_types_shapes_and_instructions(self):
        invalid = [
            {},
            {"type": "boolean", "instructions": "Approve?"},
            {"type": "choice", "instructions": "Choose", "criteria": {}},
            {"type": "choice", "instructions": "Choose", "criteria": {"only": "One"}},
            {"type": "choice", "instructions": "Choose", "criteria": ["A", "B"]},
            {"type": "score", "instructions": "Rate", "criteria": {"A": "Low", "B": "High"}},
            {"type": "score", "instructions": "Rate", "criteria": ["Only one"]},
            {"type": "noul", "instructions": "Urgent?", "criteria": {"maybe": "Unknown"}},
            {"type": "noul", "instructions": ""},
            {"type": "choice", "instructions": "Choose", "criteria": {"ref": "/private/db"}},
        ]
        for question in invalid:
            with self.subTest(question=question):
                self.graph = load_graph(EXAMPLE)
                self.approval()["question"] = question
                self.assert_invalid()

    def test_embedded_schema_errors(self):
        self.graph["input_schema"] = {"type": "not-a-type"}
        self.assert_invalid("invalid JSON Schema")

    def test_boolean_embedded_schemas_are_supported(self):
        self.graph["input_schema"] = True
        self.graph["output_schema"] = False
        self.graph["nodes"][0]["output_schema"] = True
        validate_graph(self.graph)

    def test_local_schema_references(self):
        self.graph["input_schema"] = {
            "$defs": {"text/name": {"type": "string"}},
            "properties": {"name": {"$ref": "#/$defs/text~1name"}},
        }
        validate_graph(self.graph)
        self.graph["input_schema"]["properties"]["name"]["$ref"] = "#/$defs/absent"
        self.assert_invalid("unresolved local")

    def test_schema_reference_rules_prevent_external_fetches(self):
        for schema in (
            {"$ref": "https://example.invalid/schema"},
            {"properties": {"unused": {"$ref": "https://example.invalid/schema"}}},
            {"$dynamicRef": "https://example.invalid/schema"},
            {"$id": "https://example.invalid/schema"},
            {"$schema": "http://json-schema.org/draft-07/schema#"},
        ):
            with self.subTest(schema=schema), patch("socket.create_connection", side_effect=AssertionError("network")):
                self.graph["input_schema"] = schema
                self.assert_invalid()

    def test_local_references_cannot_hide_external_or_invalid_target_schemas(self):
        for schema in (
            {"$ref": "#/definitions/entry", "definitions": {
                "entry": {"$ref": "https://example.invalid/schema"}}},
            {"$ref": "#/custom", "custom": {"$ref": "https://example.invalid/schema"}},
            {"$ref": "#/const", "const": {"$ref": "https://example.invalid/schema"}},
            {"$ref": "#/custom", "custom": {"properties": "not a schema map"}},
        ):
            with self.subTest(schema=schema), patch("socket.create_connection", side_effect=AssertionError("network")):
                self.graph["input_schema"] = schema
                self.assert_invalid()

    def test_schema_walk_does_not_treat_literal_data_as_references(self):
        self.graph["input_schema"] = {"enum": [{"$ref": "https://ordinary-data.example"}]}
        validate_graph(self.graph)

    def test_nonfinite_literals_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                self.graph["nodes"][1]["inputs"]["body"] = {"literal": value}
                self.assert_invalid("not finite JSON")

    def test_loader_rejects_duplicate_keys_and_nonfinite_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            for text in ('{"id": "first", "id": "second"}', '{"value": NaN}', '{"value": Infinity}'):
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        load_graph(path)

    def test_cli_valid_example_and_explicit_availability(self):
        result = subprocess.run(
            [sys.executable, "-m", "traceir.ir.validate", str(EXAMPLE)],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("6 nodes", result.stdout)
        self.assertIn("availability was not checked", result.stdout)
        command = [sys.executable, "-m", "traceir.ir.validate", str(EXAMPLE)]
        for operation in sorted(OPERATIONS):
            command.extend(["--operation", operation])
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("not checked", result.stdout)

    def test_cli_rejects_missing_interaction_operation(self):
        command = [sys.executable, "-m", "traceir.ir.validate", str(EXAMPLE)]
        for operation in sorted(OPERATIONS - {"request_input"}):
            command.extend(["--operation", operation])
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("unavailable operation 'request_input'", result.stderr)

    def test_cli_reports_invalid_graph_without_traceback(self):
        self.graph["entrypoint"] = "absent"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(self.graph))
            result = subprocess.run(
                [sys.executable, "-m", "traceir.ir.validate", str(path)],
                cwd=ROOT, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("unknown node ID", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


class DecideBranchTests(unittest.TestCase):
    """A DECIDE branch must be followable for every answer the node can get.

    The model chooses the answer; the graph has to say where each answer goes.
    These checks are what makes that guarantee hold before a workflow ever runs.
    """

    NOUL = {"type": "noul", "instructions": "Is the item eligible?"}
    SCORE = {"type": "score", "instructions": "How urgent?", "criteria": ["routine", "normal", "urgent"]}
    ABSENT = object()

    def setUp(self):
        self.graph = load_graph(EXAMPLE)

    def branch(self, *, question=ABSENT, on=ABSENT, routes=ABSENT, bands=ABSENT, then=ABSENT):
        node = next(n for n in self.graph["nodes"] if n["id"] == "interpret_approval")
        for name, value in (
            ("question", question), ("on", on), ("routes", routes), ("bands", bands), ("next", then),
        ):
            if value is self.ABSENT:
                continue
            if value is None:
                node.pop(name, None)
            else:
                node[name] = value
        return self

    def accept(self):
        validate_graph(self.graph)

    def reject(self, text):
        with self.assertRaises(IRValidationError) as caught:
            validate_graph(self.graph)
        self.assertIn(text, str(caught.exception))

    def test_routes_must_cover_the_declared_criteria_exactly(self):
        self.branch(routes={"APPROVE": "send_reply"}).reject("criteria without a route: DECLINE, UNCLEAR")
        self.graph = load_graph(EXAMPLE)
        self.branch(routes={"APPROVE": "send_reply", "DECLINE": "report_not_sent",
                            "UNCLEAR": "ask_approval", "MAYBE": "send_reply"}).reject(
            "routes that are not criteria: MAYBE"
        )

    def test_routes_need_a_choice_question_answered_directly(self):
        self.branch(question=self.NOUL).reject("'routes' need a 'choice' question")
        self.graph = load_graph(EXAMPLE)
        self.branch(on="confidence").reject("'routes' need a 'choice' question")

    def test_a_choice_answer_branches_with_routes_and_not_with_bands(self):
        self.branch(routes=None, bands=[{"max": 0.4, "next": "ask_approval"},
                                        {"max": None, "next": "send_reply"}]).reject(
            "branch on a 'choice' answer with 'routes', not 'bands'"
        )

    def test_bands_need_a_catch_all_last(self):
        self.branch(routes=None, question=self.NOUL,
                    bands=[{"max": 0.4, "next": "ask_approval"}]).reject("must catch everything")
        self.graph = load_graph(EXAMPLE)
        self.branch(routes=None, question=self.NOUL,
                    bands=[{"max": None, "next": "ask_approval"},
                           {"max": 0.9, "next": "send_reply"}]).reject("later bands never match")

    def test_band_maxima_must_increase(self):
        self.branch(routes=None, question=self.NOUL,
                    bands=[{"max": 0.9, "next": "ask_approval"},
                           {"max": 0.4, "next": "send_reply"},
                           {"max": None, "next": "report_not_sent"}]).reject("must increase")
        self.graph = load_graph(EXAMPLE)
        self.branch(routes=None, question=self.NOUL,
                    bands=[{"max": 0.4, "next": "ask_approval"},
                           {"max": 0.4, "next": "send_reply"},
                           {"max": None, "next": "report_not_sent"}]).reject("must increase")

    def test_band_maxima_stay_inside_the_range_the_answer_can_take(self):
        self.branch(routes=None, question=self.NOUL,
                    bands=[{"max": 4.0, "next": "report_not_sent"},
                           {"max": None, "next": "send_reply"}]).reject("0..1 range")
        self.graph = load_graph(EXAMPLE)
        self.branch(routes=None, question=self.SCORE,
                    bands=[{"max": 0, "next": "ask_approval"},
                           {"max": 7, "next": "send_reply"},
                           {"max": None, "next": "report_not_sent"}]).reject("0..2 range")
        self.graph = load_graph(EXAMPLE)
        self.branch(routes=None, question=self.SCORE,
                    bands=[{"max": 0, "next": "ask_approval"},
                           {"max": 2, "next": "send_reply"},
                           {"max": None, "next": "report_not_sent"}]).accept()

    def test_confidence_bands_may_use_any_question_type(self):
        for question in (self.NOUL, self.SCORE, {"type": "choice", "instructions": "Which?",
                                                 "criteria": {"A": "a", "B": "b"}}):
            with self.subTest(question_type=question["type"]):
                self.graph = load_graph(EXAMPLE)
                self.branch(question=question, on="confidence", routes=None,
                            bands=[{"max": 0.6, "next": "ask_approval"},
                                   {"max": None, "next": "send_reply"}]).accept()

    def test_certainty_without_a_branch_is_a_mistake(self):
        self.branch(routes=None, then="send_reply", on="confidence").reject("has no effect")
        self.graph = load_graph(EXAMPLE)
        self.branch(routes=None, then="send_reply").accept()

    def test_band_targets_are_checked_as_successors(self):
        self.branch(routes=None, question=self.NOUL,
                    bands=[{"max": 0.4, "next": "nowhere"},
                           {"max": None, "next": "send_reply"}]).reject("unknown successor 'nowhere'")

    def test_criteria_from_state_leave_the_labels_to_the_run(self):
        self.branch(question={"type": "choice", "instructions": "Which?",
                             "criteria": {"ref": "/state/kinds"}},
                    routes={"WHATEVER": "send_reply"}).accept()


if __name__ == "__main__":
    unittest.main()
