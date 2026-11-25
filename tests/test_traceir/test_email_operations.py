"""Tests for the reviewed EXECUTE helpers of the email-reply workflow.

These check code against the artifact: names, signatures, routing outcomes, and
result contracts come from `graph.json`, so graph and code cannot drift
silently.
"""

import inspect
import json
from pathlib import Path
import unittest
from typing import Any
from unittest.mock import patch

from jsonschema import Draft202012Validator

from traceir.ir.models import ExecuteNode
from traceir.ir.validate import load_graph, validate_graph
from workflows.email_reply.operations import (
    OUTBOX,
    WORKFLOW_OPERATIONS,
    prepare_review,
    report_not_sent,
    request_input,
    send_reply,
)

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "workflows/email_reply/graph.json"


class EmailOperationTests(unittest.TestCase):
    def setUp(self):
        self.graph = load_graph(GRAPH)
        self.workflow = validate_graph(self.graph, operations=set(WORKFLOW_OPERATIONS))

    def test_operation_bindings_cover_every_execute_node(self):
        referenced = {
            node.operation
            for node in self.workflow.nodes
            if isinstance(node, ExecuteNode)
        }
        self.assertEqual(referenced, set(WORKFLOW_OPERATIONS))

    def test_signatures_match_declared_inputs(self):
        for node in self.workflow.nodes:
            if not isinstance(node, ExecuteNode) or node.operation not in WORKFLOW_OPERATIONS:
                continue
            with self.subTest(node=node.id):
                signature = inspect.signature(WORKFLOW_OPERATIONS[node.operation])
                self.assertEqual(set(signature.parameters), set(node.inputs))
                for parameter in signature.parameters.values():
                    self.assertIs(parameter.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_helpers_are_plain_functions_of_the_example_package(self):
        # Import independence is checked statically in test_package_boundaries;
        # what matters here is that these are the package's own plain functions.
        for name, operation in WORKFLOW_OPERATIONS.items():
            with self.subTest(operation=name):
                module = inspect.getmodule(operation)
                assert module is not None
                self.assertEqual(module.__name__, "workflows.email_reply.operations")
                self.assertEqual(operation.__name__, name)

    def test_returned_values_are_plain_json(self):
        results = [
            prepare_review(recipient="marta@example.com", body="a draft"),
            report_not_sent(),
            send_reply(in_reply_to="email_123", recipient="marta@example.com", body="a draft"),
        ]
        for result in results:
            with self.subTest(result=result):
                json.dumps(result, allow_nan=False)

    def test_prepare_review_reproduces_the_reviewed_values(self):
        recipient = "marta@example.com"
        body = "Hi Marta,\n\nTuesday at 15:00 works for me."
        result = prepare_review(recipient=recipient, body=body)
        self.assertEqual(
            result,
            f"To: {recipient}\n\nDraft reply:\n\n{body}\n\n"
            "Send this reply? Answer yes to send it, or no to skip it.",
        )
        self.assertIn(body, result)

    def test_request_input_shows_the_message_and_returns_the_reply(self):
        shown = []
        with patch("builtins.input", return_value="Yes, send it.") as typed, patch(
            "builtins.print", side_effect=shown.append
        ):
            reply = request_input(message="Send this reply?")
        self.assertEqual(reply, "Yes, send it.")
        self.assertEqual(shown, ["\nSend this reply?"])
        typed.assert_called_once_with("> ")

    def test_prepare_review_refuses_empty_inputs(self):
        invalid: list[tuple[Any, Any]] = [
            ("", "a draft"),
            ("marta@example.com", ""),
            (None, "a draft"),
            ("marta@example.com", None),
            (1, "a draft"),
        ]
        for recipient, body in invalid:
            with self.subTest(recipient=recipient, body=body):
                with self.assertRaises(ValueError):
                    prepare_review(recipient=recipient, body=body)

    def test_report_not_sent_matches_the_graph_output_contract(self):
        Draft202012Validator(self.graph["output_schema"]).validate(report_not_sent())

    def test_send_reply_records_the_approved_message_untouched(self):
        before = len(OUTBOX)
        message = {
            "in_reply_to": "email_123",
            "recipient": "marta@example.com",
            "body": "Hi Marta,\n\nTuesday works.",
        }
        result = send_reply(**message)
        self.assertEqual(OUTBOX[before], message)
        Draft202012Validator(self.graph["output_schema"]).validate(result)
        self.assertEqual(result["message_id"], f"message-{before + 1}")


if __name__ == "__main__":
    unittest.main()
