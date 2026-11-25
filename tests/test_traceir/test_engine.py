"""Offline engine tests using stub operations, generators, and deciders.

Nothing here calls a model: GENERATE and DECIDE are given stubs, and the
operations that would speak to a person are replaced by one that answers from a
script.
"""

from pathlib import Path
import unittest
from typing import Any

from traceir.engine.core import Completed, Failed, run
from traceir.ir.models import Workflow
from traceir.ir.validate import load_graph, validate_graph
from workflows.email_reply.operations import WORKFLOW_OPERATIONS

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "workflows/email_reply/graph.json"
INPUT = {
    "email": {"id": "email_123", "sender": "marta@example.com", "body": "Tuesday?"},
    "instruction": "Confirm the time.",
}
DRAFT = "Sure, Tuesday works."
REVIEWED = "Yes, please send it."


def completed(result) -> Completed:
    assert isinstance(result, Completed), result
    return result


def failed(result) -> Failed:
    assert isinstance(result, Failed), result
    return result


class StubHost:
    """The package's terminal operations, replaced.

    `request_input` answers from a script instead of waiting for typing, so a test
    can see how many times the workflow asked and what it asked each time.
    """

    def __init__(self, replies: list[str]):
        self.replies = replies
        self.messages: list[str] = []
        self.sent: list[dict] = []

    def request_input(self, *, message: str) -> str:
        self.messages.append(message)
        if not self.replies:
            raise AssertionError("the workflow asked more times than the script answers")
        return self.replies.pop(0)

    def send_reply(self, *, in_reply_to: str, recipient: str, body: str) -> dict:
        self.sent.append({"in_reply_to": "email_123", "recipient": recipient, "body": body})
        return {"status": "submitted", "message_id": f"outbox-{len(self.sent)}"}


def scripted_decider(*labels: str):
    """Answer the node's one question with each label, in order."""

    remaining = list(labels)

    def decide(node, inputs, question):
        return {"type": question["type"], "choice": remaining.pop(0)}

    return decide


def run_example(
    decide,
    *,
    replies: list[str] | None = None,
    generate=lambda node, inputs: DRAFT,
    operations: dict | None = None,
    graph: dict | None = None,
    initial_input: dict = INPUT,
    max_steps: int = 50,
):
    """Run the email example with a stubbed host, returning the result and what it did."""
    host = StubHost([REVIEWED] if replies is None else list(replies))
    bound = {
        **WORKFLOW_OPERATIONS,
        "request_input": host.request_input,
        "send_reply": host.send_reply,
        **(operations or {}),
    }
    workflow = validate_graph(
        load_graph(GRAPH) if graph is None else graph, operations=set(bound)
    )
    result = run(
        workflow,
        initial_input,
        operations=bound,
        generate=generate,
        decide=decide,
        max_steps=max_steps,
    )
    return result, host


class ExampleWorkflowTests(unittest.TestCase):
    def test_approved_reply_is_sent_once_after_review(self):
        result, host = run_example(scripted_decider("APPROVE"))
        self.assertEqual(completed(result).output, {"status": "submitted", "message_id": "outbox-1"})
        self.assertEqual(
            host.sent,
            [{"in_reply_to": "email_123", "recipient": "marta@example.com", "body": DRAFT}],
        )
        self.assertEqual(
            [node_id for node_id, _ in result.trace],
            [
                "draft_reply",
                "prepare_review",
                "ask_approval",
                "interpret_approval",
                "send_reply",
            ],
        )

    def test_state_holds_declared_outputs(self):
        result, _ = run_example(scripted_decider("APPROVE"))
        state = completed(result).state
        self.assertEqual(set(state), {"draft", "review_message", "approval_reply", "approval"})
        self.assertEqual(state["draft"], DRAFT)
        self.assertEqual(state["approval_reply"], REVIEWED)
        self.assertEqual(state["approval"], {"type": "choice", "choice": "APPROVE"})
        self.assertIn(DRAFT, state["review_message"])
        self.assertIn("marta@example.com", state["review_message"])

    def test_declined_reply_never_touches_the_mail_tool(self):
        result, host = run_example(scripted_decider("DECLINE"))
        self.assertEqual(completed(result).output, {"status": "not_sent", "message_id": None})
        self.assertEqual(host.sent, [])

    def test_unclear_reply_asks_again_about_the_same_draft(self):
        result, host = run_example(
            scripted_decider("UNCLEAR", "APPROVE"),
            replies=["Hmm, maybe.", "Yes."],
        )
        self.assertEqual(completed(result).output["status"], "submitted")
        self.assertEqual(host.messages, host.messages[:1] * 2)
        self.assertIn(DRAFT, host.messages[0])
        self.assertEqual(len(host.sent), 1)

    def test_each_visit_to_a_node_runs_its_operation_once(self):
        calls: list[str] = []
        result, _ = run_example(
            scripted_decider("APPROVE"),
            operations={"prepare_review": lambda **inputs: calls.append("prepare") or "reviewed"},
        )
        self.assertIsInstance(result, Completed)
        self.assertEqual(calls, ["prepare"])
        self.assertEqual([node_id for node_id, _ in result.trace].count("prepare_review"), 1)

    def test_unclear_loop_is_stopped_by_the_step_limit(self):
        result, host = run_example(
            scripted_decider(*["UNCLEAR"] * 40),
            replies=["Keep asking."] * 40,
            max_steps=8,
        )
        self.assertIn("step limit of 8", failed(result).detail)
        self.assertEqual(host.sent, [])

    def test_a_failed_run_reports_the_state_it_reached(self):
        result, _ = run_example(scripted_decider("APPROVE"), generate=lambda n, i: 42)
        self.assertIn("'draft_reply' value does not match its schema", failed(result).detail)
        self.assertEqual(failed(result).state, {})

        result, _ = run_example(scripted_decider("MAYBE"))
        self.assertIn("answer does not match its schema", failed(result).detail)
        self.assertEqual(
            set(failed(result).state),
            {"draft", "review_message", "approval_reply"},
        )

    def test_the_trace_stops_at_the_last_node_that_completed(self):
        """A node the engine rejected is absent: it did not happen as far as this run goes."""
        result, _ = run_example(scripted_decider("MAYBE"))
        self.assertEqual(
            [node_id for node_id, _ in failed(result).trace],
            ["draft_reply", "prepare_review", "ask_approval"],
        )

    def test_unbound_operation_fails_before_anything_runs(self):
        workflow = validate_graph(load_graph(GRAPH))
        bound = {name: op for name, op in WORKFLOW_OPERATIONS.items() if name != "request_input"}
        with self.assertRaisesRegex(ValueError, "unbound operations: request_input"):
            run(workflow, INPUT, operations=bound)

    def test_input_is_validated_against_the_declared_contract(self):
        with self.assertRaisesRegex(ValueError, "initial input does not match its schema"):
            run_example(scripted_decider("APPROVE"), initial_input={"email": {"id": "email_123"}})

    def test_decide_answer_is_checked_against_the_declared_criteria(self):
        result, _ = run_example(scripted_decider("MAYBE"))
        detail = failed(result).detail
        self.assertIn("'interpret_approval' answer does not match its schema", detail)
        self.assertIn("MAYBE", detail)
        self.assertIn("APPROVE", detail)

    def test_a_decide_answer_must_carry_an_answer(self):
        result, _ = run_example(lambda node, inputs, question: {"type": "choice"})
        self.assertIn("'choice' is a required property", failed(result).detail)

    def test_generated_values_are_checked_against_the_node_schema(self):
        result, _ = run_example(scripted_decider("APPROVE"), generate=lambda n, i: 42)
        self.assertIn("'draft_reply' value does not match its schema", failed(result).detail)

    def test_final_output_is_checked_against_the_output_schema(self):
        result, _ = run_example(
            scripted_decider("DECLINE"), operations={"report_not_sent": lambda: {"status": "maybe"}}
        )
        self.assertIn("workflow output does not match its schema", failed(result).detail)

    def test_unresolved_reference_fails_with_its_pointer(self):
        graph = load_graph(GRAPH)
        graph["nodes"][0]["inputs"]["instruction"] = {"ref": "/state/absent"}
        result, _ = run_example(scripted_decider("APPROVE"), graph=graph)
        self.assertIn("unresolved reference '/state/absent'", failed(result).detail)

    def test_a_routed_decide_answer_is_still_stored(self):
        """The answer is data for later nodes, not only a branch selector."""
        result, _ = run_example(scripted_decider("APPROVE"))
        self.assertEqual(completed(result).state["approval"], {"type": "choice", "choice": "APPROVE"})

    def test_a_node_without_a_backend_is_reported(self):
        result, _ = run_example(scripted_decider("APPROVE"), generate=None)
        self.assertIn("no GENERATE backend is configured", failed(result).detail)
        result, _ = run_example(None)
        self.assertIn("no DECIDE backend is configured", failed(result).detail)


CHOICE = {
    "type": "choice",
    "instructions": "Did the customer approve?",
    "criteria": {"APPROVE": "yes", "DECLINE": "no", "UNCLEAR": "neither"},
}
SCORE = {
    "type": "score",
    "instructions": "How urgent?",
    "criteria": ["routine", "normal", "urgent"],
}
NOUL = {"type": "noul", "instructions": "Is the item eligible?"}


class DecideRoutingTests(unittest.TestCase):
    """Each DECIDE shape, routing on the answer or on confidence.

    These pin the branch the engine takes for a given answer, so a node cannot
    silently read the wrong number or fall through to the wrong successor.
    """

    def target(
        self, question: dict, continuation: dict, answer: dict, *, seed: Any = None
    ) -> Any:
        terminals = sorted(
            {
                value
                for key, value in (continuation.get("routes") or {}).items()
                if value is not None
            }
            | {band["next"] for band in continuation.get("bands") or [] if band["next"]}
            | ({continuation["next"]} if "next" in continuation else set())
        )
        nodes: list[dict[str, Any]] = []
        operations: dict[str, Any] = {}
        entrypoint = "decide"
        if seed is not None:
            # State only grows through a node, so a test seeds it with a node too.
            entrypoint = "seed"
            nodes.append(
                {
                    "id": "seed",
                    "type": "EXECUTE",
                    "operation": "seed",
                    "inputs": {},
                    "write_to": "/state/kinds",
                    "next": "decide",
                }
            )
            operations["seed"] = lambda value=seed: value
        nodes.append(
            {
                "id": "decide",
                "type": "DECIDE",
                "inputs": {},
                "question": question,
                "write_to": "/state/answer",
                **continuation,
            }
        )
        for name in terminals:
            nodes.append(
                {"id": name, "type": "EXECUTE", "operation": name, "inputs": {}, "next": None}
            )
            operations[name] = (lambda name: (lambda: name))(name)
        workflow = validate_graph(
            {
                "schema_version": "0.1",
                "entrypoint": entrypoint,
                "input_schema": {"type": "object"},
                "output_schema": {"type": ["string", "object"]},
                "nodes": nodes,
            },
            operations=set(operations),
        )

        def decide(node, inputs, resolved_question):
            return {**answer, "type": resolved_question["type"]}

        result = run(workflow, {}, operations=operations, decide=decide)
        if isinstance(result, Failed):
            raise AssertionError(result.detail)
        return result.output

    def test_choice_routes_on_the_selected_label(self):
        routes = {"APPROVE": "send", "DECLINE": "skip", "UNCLEAR": "ask"}
        for label, expected in (
            ("APPROVE", "send"),
            ("DECLINE", "skip"),
            ("UNCLEAR", "ask"),
        ):
            with self.subTest(label=label):
                answer = {"choice": label, "confidence": 0.99}
                self.assertEqual(self.target(CHOICE, {"routes": routes}, answer), expected)

    def test_score_routes_on_its_level(self):
        bands = [
            {"max": 0, "next": "auto_refund"},
            {"max": 1, "next": "standard"},
            {"max": None, "next": "escalate"},
        ]
        for level, expected in ((0, "auto_refund"), (1, "standard"), (2, "escalate")):
            with self.subTest(level=level):
                answer = {"score": level, "confidence": 0.9}
                self.assertEqual(self.target(SCORE, {"bands": bands}, answer), expected)

    def test_noul_routes_on_its_probability(self):
        bands = [
            {"max": 0.35, "next": "reject"},
            {"max": 0.70, "next": "ask_user"},
            {"max": None, "next": "approve"},
        ]
        for probability, expected in ((0.10, "reject"), (0.50, "ask_user"), (0.92, "approve")):
            with self.subTest(probability=probability):
                answer = {"noul": probability, "confidence": 0.9}
                self.assertEqual(self.target(NOUL, {"bands": bands}, answer), expected)

    def test_bands_on_confidence_ignore_the_selected_label(self):
        bands = [{"max": 0.60, "next": "ask_user"}, {"max": None, "next": "send"}]
        for label, confidence, expected in (
            ("APPROVE", 0.20, "ask_user"),
            ("APPROVE", 0.95, "send"),
            ("DECLINE", 0.95, "send"),
        ):
            with self.subTest(label=label, confidence=confidence):
                answer = {"choice": label, "confidence": confidence}
                self.assertEqual(
                    self.target(CHOICE, {"on": "confidence", "bands": bands}, answer),
                    expected,
                )

    def test_a_band_may_end_the_workflow(self):
        bands = [{"max": 0.4, "next": None}, {"max": None, "next": "apply"}]
        answer = {"noul": 0.9, "confidence": 0.2}
        # The terminating node's value is the answer itself, as the engine stores it.
        self.assertEqual(
            self.target(NOUL, {"on": "confidence", "bands": bands}, answer),
            {**answer, "type": "noul"},
        )

    def test_a_bare_decision_continues_without_branching(self):
        self.assertEqual(self.target(CHOICE, {"next": "after"}, {"choice": "DECLINE"}), "after")

    def test_an_unrouted_label_fails_rather_than_choosing_a_default(self):
        # Legal at load time: criteria read from state name labels the graph cannot know.
        question = {"type": "choice", "instructions": "Which?", "criteria": {"ref": "/state/kinds"}}
        with self.assertRaisesRegex(AssertionError, "answered 'B', which is not routed"):
            self.target(
                question,
                {"routes": {"A": "a"}},
                {"choice": "B"},
                seed={"A": "first", "B": "second"},
            )

    def test_an_answer_without_certainty_cannot_branch_on_it(self):
        bands = [{"max": 0.60, "next": "ask_user"}, {"max": None, "next": "send"}]
        with self.assertRaisesRegex(AssertionError, "'confidence' is a required property"):
            self.target(CHOICE, {"on": "confidence", "bands": bands}, {"choice": "APPROVE"})


class ExecuteRoutingTests(unittest.TestCase):
    """Routing and completion on a small graph, independent of the example."""

    def workflow(self) -> Workflow:
        return validate_graph(
            {
                "schema_version": "0.1",
                "entrypoint": "choose",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "string"},
                "nodes": [
                    {
                        "id": "choose",
                        "type": "EXECUTE",
                        "operation": "choose",
                        "inputs": {},
                        "routes": {"BIG": "big", "SMALL": None},
                    },
                    {
                        "id": "big",
                        "type": "EXECUTE",
                        "operation": "big",
                        "inputs": {},
                        "next": None,
                    },
                ],
            },
            operations={"choose", "big"},
        )

    def run_with(self, result: str):
        calls: list[str] = []

        def choose() -> str:
            calls.append("choose")
            return result

        outcome = run(
            self.workflow(),
            {},
            operations={"choose": choose, "big": lambda: "finished"},
        )
        return outcome, calls

    def test_route_to_node_and_to_termination(self):
        outcome, _ = self.run_with("BIG")
        self.assertEqual(completed(outcome).output, "finished")
        outcome, _ = self.run_with("SMALL")
        self.assertEqual(completed(outcome).output, "SMALL")

    def test_undeclared_route_value_fails(self):
        outcome, _ = self.run_with("HUGE")
        self.assertIn("HUGE", failed(outcome).detail)

    def test_a_route_name_is_not_stored_as_a_node_value(self):
        outcome, _ = self.run_with("BIG")
        self.assertNotIn("BIG", completed(outcome).state.values())

    def test_a_routed_operation_runs_once(self):
        outcome, calls = self.run_with("BIG")
        self.assertEqual(calls, ["choose"])
        self.assertEqual([node_id for node_id, _ in outcome.trace], ["choose", "big"])


if __name__ == "__main__":
    unittest.main()


class LifecycleHookTests(unittest.TestCase):
    """`setup` prepares a resource, `teardown` runs whatever happened."""

    def lifecycle(self, *, setup=None, teardown=None, decide=None):
        calls: list[str] = []

        def wrapped_setup(initial_input):
            calls.append(f"setup:{initial_input['email']['id']}")
            if setup is not None:
                setup(initial_input)

        def wrapped_teardown():
            calls.append("teardown")
            if teardown is not None:
                teardown()

        host = StubHost([REVIEWED])
        bound = {
            **WORKFLOW_OPERATIONS,
            "request_input": host.request_input,
            "send_reply": host.send_reply,
        }
        result = run(
            validate_graph(load_graph(GRAPH), operations=set(bound)),
            INPUT,
            operations=bound,
            generate=lambda node, inputs: DRAFT,
            decide=decide or scripted_decider("APPROVE"),
            setup=wrapped_setup,
            teardown=wrapped_teardown,
        )
        return result, calls

    def test_setup_sees_the_initial_input_and_runs_before_the_first_node(self):
        result, calls = self.lifecycle()
        self.assertIsInstance(result, Completed)
        self.assertEqual(calls[0], "setup:email_123")

    def test_teardown_runs_after_a_completed_run(self):
        _, calls = self.lifecycle()
        self.assertEqual(calls[-1], "teardown")

    def test_teardown_runs_after_a_failed_run(self):
        result, calls = self.lifecycle(
            decide=lambda node, inputs, question: {"type": "choice", "choice": "MAYBE"}
        )
        self.assertIsInstance(result, Failed)
        self.assertEqual(calls[-1], "teardown")

    def test_a_failing_setup_is_a_result_not_a_raise_and_teardown_still_runs(self):
        def broken(initial_input):
            raise RuntimeError("no environment")

        result, calls = self.lifecycle(setup=broken)
        failure = failed(result)
        self.assertIn("no environment", failure.detail)
        self.assertEqual(failure.trace, [])
        self.assertEqual(calls[-1], "teardown")

    def test_a_failing_teardown_propagates_because_the_record_is_unfinished(self):
        def broken():
            raise RuntimeError("cannot write the run record")

        with self.assertRaisesRegex(RuntimeError, "cannot write the run record"):
            self.lifecycle(teardown=broken)

    def test_a_package_without_hooks_runs_unchanged(self):
        result, _ = run_example(scripted_decider("APPROVE"))
        self.assertIsInstance(result, Completed)
