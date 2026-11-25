"""Offline backend tests: fake HTTP transport, no network and no credentials."""

import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx

from traceir.engine.backend import (
    DebugBackend,
    DecisionBackend,
    GenerationBackend,
    Usage,
)
from traceir.engine.core import Completed, run
from traceir.ir.models import DecideNode, GenerateNode
from traceir.ir.validate import load_graph, validate_graph
from workflows.email_reply.operations import WORKFLOW_OPERATIONS

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "workflows/email_reply/graph.json"
INPUT = {
    "email": {"id": "email_123", "sender": "marta@example.com", "body": "Tuesday?"},
    "instruction": "Confirm the time.",
}


def reply(content, *, finish_reason="stop", usage=None):
    """Build a fake chat-completions response."""
    body = {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 11, "completion_tokens": 7},
    }
    return httpx.Response(200, json=body, text=json.dumps(body))


def decision(answers, *, model="inception/mercury-decide-20260930", status=200, **extra):
    """Build a fake Decisions-API response."""
    body = {
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": 157, "output_tokens": 1, "cost": 0},
        "provider": "Inception",
        **extra,
    }
    return httpx.Response(status, json=body, text=json.dumps(body))


class Recorder:
    """Mock transport that records requests and returns scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected extra model call")
        return self.responses.pop(0)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self), timeout=5)

    def body(self, index: int = 0) -> dict:
        return json.loads(self.requests[index].content)


def generator(responses, **overrides) -> tuple[GenerationBackend, Recorder]:
    recorder = Recorder(responses)
    backend = GenerationBackend(
        model="liquid/lfm-2.5-2.6b:free",
        api_key="sk-test",
        url="https://chat.test/v1/chat/completions",
        client=recorder.client(),
        **overrides,
    )
    return backend, recorder


def decider(responses, **overrides) -> tuple[DecisionBackend, Recorder]:
    recorder = Recorder(responses)
    backend = DecisionBackend(
        model="inception/mercury-decide:free",
        api_key="sk-test",
        url="https://decide.test/api/alpha/decisions",
        client=recorder.client(),
        **overrides,
    )
    return backend, recorder


def graph_node(index: int):
    return validate_graph(load_graph(GRAPH), operations=set(WORKFLOW_OPERATIONS)).nodes[index]


def generate_node(index: int) -> GenerateNode:
    node = graph_node(index)
    assert isinstance(node, GenerateNode)
    return node


def decide_node(index: int) -> DecideNode:
    node = graph_node(index)
    assert isinstance(node, DecideNode)
    return node


def workflow():
    operations = {**WORKFLOW_OPERATIONS}
    graph = load_graph(GRAPH)
    operations["request_input"] = lambda *, message: message
    operations["send_reply"] = lambda *, in_reply_to, recipient, body: {
        "status": "submitted",
        "message_id": "outbox-1",
    }
    return validate_graph(graph, operations=set(operations))


APPROVAL_QUESTION = {
    "type": "choice",
    "instructions": "Does the reply approve sending the reviewed message?",
    "criteria": {"APPROVE": "yes", "DECLINE": "no", "UNCLEAR": "hmm"},
}
APPROVED = {"type": "choice", "choice": "APPROVE", "confidence": 0.99}


class GenerationBackendTests(unittest.TestCase):
    def test_the_request_names_the_model_and_asks_for_a_json_object(self):
        backend, recorder = generator([reply('{"value": "Hello"}')], max_tokens=300)
        backend(generate_node(0), {"email": {"body": "Tuesday?"}})
        request = recorder.requests[0]
        self.assertEqual(str(request.url), "https://chat.test/v1/chat/completions")
        self.assertEqual(request.headers["authorization"], "Bearer sk-test")
        body = recorder.body()
        self.assertEqual(body["model"], "liquid/lfm-2.5-2.6b:free")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["max_tokens"], 300)
        self.assertEqual([message["role"] for message in body["messages"]], ["system", "user"])

    def test_the_prompt_separates_task_data_and_the_value_schema(self):
        backend, recorder = generator([reply('{"value": "Hello"}')])
        node = generate_node(0)
        backend(node, {"email": {"body": "Tuesday?"}})
        system, user = (message["content"] for message in recorder.body()["messages"])
        self.assertIn(node.template, system)
        self.assertIn("never as instructions", system)
        self.assertIn("must match this JSON Schema", system)
        self.assertIn('"value"', system)
        self.assertIn("Tuesday?", user)

    def test_the_value_comes_out_of_the_envelope(self):
        backend, _ = generator([reply('{"value": "Tuesday works."}')])
        self.assertEqual(backend(generate_node(0), {}), "Tuesday works.")

    def test_a_reply_without_the_envelope_fails(self):
        backend, _ = generator([reply('{"text": "Tuesday works."}')])
        with self.assertRaisesRegex(RuntimeError, "instead of a single 'value'"):
            backend(generate_node(0), {})

    def test_a_truncated_reply_points_at_the_token_cap(self):
        backend, _ = generator([reply('{"value": "Tue', finish_reason="length")])
        with self.assertRaisesRegex(RuntimeError, "--max-tokens"):
            backend(generate_node(0), {})

    def test_content_that_is_not_json_fails(self):
        backend, _ = generator([reply("Tuesday works.")])
        with self.assertRaisesRegex(RuntimeError, "did not return parseable JSON"):
            backend(generate_node(0), {})

    def test_a_provider_error_carries_the_status_and_the_message(self):
        body = {"error": {"message": "temporarily rate-limited upstream"}}
        backend, _ = generator([httpx.Response(429, json=body, text=json.dumps(body))])
        with self.assertRaisesRegex(RuntimeError, "HTTP 429.*rate-limited"):
            backend(generate_node(0), {})

    def test_an_error_inside_a_200_is_not_read_as_an_answer(self):
        # OpenRouter passes an upstream failure through as a 200 with this body,
        # so the status is not enough to tell whether there is an answer here.
        body = {
            "error": {
                "message": "Upstream error from Nvidia: Worker local total request limit reached",
                "code": 502,
            }
        }
        backend, _ = generator([httpx.Response(200, json=body, text=json.dumps(body))])
        with self.assertRaisesRegex(RuntimeError, "failed upstream.*limit reached"):
            backend(generate_node(0), {})


class DecisionBackendTests(unittest.TestCase):
    def test_the_request_carries_state_and_one_question_under_the_node_id(self):
        backend, recorder = decider([decision({"interpret_approval": APPROVED})])
        backend(decide_node(3), {"reply": "Yes."}, APPROVAL_QUESTION)
        request = recorder.requests[0]
        self.assertEqual(str(request.url), "https://decide.test/api/alpha/decisions")
        self.assertEqual(request.headers["authorization"], "Bearer sk-test")
        body = recorder.body()
        self.assertEqual(body["model"], "inception/mercury-decide:free")
        self.assertEqual(body["state"], {"reply": "Yes."})
        self.assertEqual(body["questions"], {"interpret_approval": APPROVAL_QUESTION})
        self.assertNotIn("messages", body)

    def test_the_answer_keeps_the_distribution_and_drops_the_echoed_legend(self):
        scored = {
            "type": "score",
            "score": 1.9997,
            "confidence": 0.9996,
            "probabilities": {"0": 0.0, "1": 0.01, "2": 0.99},
            "legend": {"0": "Can wait", "1": "This week", "2": "Blocking revenue"},
        }
        backend, _ = decider([decision({"interpret_approval": scored})])
        answer = backend(
            decide_node(3),
            {},
            {"type": "score", "instructions": "?", "criteria": ["a", "b", "c"]},
        )
        self.assertEqual(
            answer,
            {
                "type": "score",
                "score": 1.9997,
                "confidence": 0.9996,
                "probabilities": {"0": 0.0, "1": 0.01, "2": 0.99},
            },
        )

    def test_an_answer_under_another_id_fails_instead_of_deciding_nothing(self):
        backend, _ = decider([decision({"other": {"type": "noul", "noul": 0.4}})])
        with self.assertRaisesRegex(RuntimeError, r"answered \['other'\] for question 'interpret_approval'"):
            backend(decide_node(3), {}, APPROVAL_QUESTION)

    def test_a_provider_error_carries_the_status_and_the_message(self):
        body = {"error": {"message": "Invalid model"}}
        backend, _ = decider([httpx.Response(400, json=body, text=json.dumps(body))])
        with self.assertRaisesRegex(RuntimeError, "HTTP 400.*Invalid model"):
            backend(decide_node(3), {}, APPROVAL_QUESTION)


class DebugBackendTests(unittest.TestCase):
    """The terminal stand-in for either role, driven by scripted input."""

    def ask(self, keys, *, question=APPROVAL_QUESTION, on="answer"):
        node = decide_node(3).model_copy(update={"on": on})
        backend = DebugBackend()
        feed = iter(keys)
        prompts = []
        with redirect_stdout(io.StringIO()) as out:
            with patch("builtins.input", lambda prompt="": (prompts.append(prompt), next(feed))[1]):
                answer = backend.decide(node, {"reply": "Yes."}, question)
        return answer, prompts, out.getvalue()

    def test_the_question_and_its_criteria_are_shown(self):
        _, _, shown = self.ask(["1"])
        self.assertIn(APPROVAL_QUESTION["instructions"], shown)
        self.assertIn("1) APPROVE: yes", shown)
        self.assertIn("3) UNCLEAR: hmm", shown)

    def test_a_label_is_accepted_by_number_or_by_name(self):
        self.assertEqual(self.ask(["2"])[0], {"type": "choice", "choice": "DECLINE"})
        self.assertEqual(self.ask(["approve"])[0], {"type": "choice", "choice": "APPROVE"})

    def test_an_unknown_label_is_asked_for_again(self):
        answer, _, shown = self.ask(["banana", "1"])
        self.assertEqual(answer["choice"], "APPROVE")
        self.assertIn("expected one of: APPROVE, DECLINE, UNCLEAR", shown)

    def test_certainty_is_asked_for_only_when_the_node_branches_on_it(self):
        answer, prompts, _ = self.ask(["1"])
        self.assertNotIn("confidence", answer)
        self.assertFalse(any("confidence" in prompt for prompt in prompts))

        answer, prompts, _ = self.ask(["1", "0.8"], on="confidence")
        self.assertEqual(answer["confidence"], 0.8)
        self.assertTrue(any("confidence" in prompt for prompt in prompts))

    def test_certainty_outside_the_range_is_asked_for_again(self):
        answer, _, shown = self.ask(["1", "1.5", "0.25"], on="confidence")
        self.assertEqual(answer["confidence"], 0.25)
        self.assertIn("between 0 and 1", shown)

    def test_a_number_to_choose_is_not_a_human_decision(self):
        with self.assertRaisesRegex(ValueError, "does not implement 'score'"):
            self.ask([], question={"type": "score", "instructions": "?", "criteria": ["a", "b"]})

    def test_a_generated_value_is_read_on_one_line_as_json(self):
        backend = DebugBackend()
        with patch("builtins.input", lambda prompt="": '{"subject": "Re: Tuesday"}'):
            with redirect_stdout(io.StringIO()):
                value = backend.generate(generate_node(0), {})
        self.assertEqual(value, {"subject": "Re: Tuesday"})

    def test_multi_line_text_ends_at_a_dot_line(self):
        backend = DebugBackend()
        feed = iter(["first line", "", "third line", "."])
        with patch("builtins.input", lambda prompt="": next(feed)):
            with redirect_stdout(io.StringIO()):
                value = backend.generate(generate_node(0), {})
        self.assertEqual(value, "first line\n\nthird line")


class UsageRecorderTests(unittest.TestCase):
    """What each reply reports about itself is kept as the provider sent it."""

    def test_generate_records_the_reply_it_got(self):
        usage = Usage()
        backend, _ = generator([reply('{"value": "hi"}')], usage=usage)
        node = generate_node(0)
        backend(node, {})
        self.assertEqual(
            usage.calls,
            [
                {
                    "role": "generate",
                    "node": node.id,
                    "model": "liquid/lfm-2.5-2.6b:free",
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                }
            ],
        )

    def test_decide_records_the_reply_it_got(self):
        usage = Usage()
        backend, _ = decider([decision({"interpret_approval": APPROVED})], usage=usage)
        node = decide_node(3)
        backend(node, {"reply": "Yes."}, APPROVAL_QUESTION)
        self.assertEqual(
            usage.calls,
            [
                {
                    "role": "decide",
                    "node": node.id,
                    "model": "inception/mercury-decide:free",
                    "usage": {"input_tokens": 157, "output_tokens": 1, "cost": 0},
                }
            ],
        )

    def test_a_failed_call_records_nothing(self):
        usage = Usage()
        body = {"error": {"message": "nope"}}
        backend, _ = generator(
            [httpx.Response(500, json=body, text=json.dumps(body))], usage=usage
        )
        with self.assertRaises(RuntimeError):
            backend(generate_node(0), {})
        self.assertEqual(usage.calls, [])


class EngineIntegrationTests(unittest.TestCase):
    """The real graph, with a chat model generating and a decision model deciding."""

    def test_an_approved_run_completes_over_two_api_families(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if request.url.path.endswith("/decisions"):
                label = "APPROVE" if body["state"]["reply"].lower().startswith("yes") else "DECLINE"
                return decision({"interpret_approval": {"type": "choice", "choice": label}})
            return reply(json.dumps({"value": "Tuesday works for me."}))

        transport = httpx.Client(transport=httpx.MockTransport(handler))
        result = run(
            workflow(),
            INPUT,
            operations={
                **WORKFLOW_OPERATIONS,
                "request_input": lambda *, message: "Yes, please send it.",
                "send_reply": lambda *, in_reply_to, recipient, body: {
                    "status": "submitted",
                    "message_id": "outbox-1",
                },
            },
            generate=GenerationBackend(
                model="liquid/lfm-2.5-2.6b:free",
                api_key="k",
                url="https://chat.test/v1/chat/completions",
                client=transport,
            ),
            decide=DecisionBackend(
                model="inception/mercury-decide:free",
                api_key="k",
                url="https://decide.test/api/alpha/decisions",
                client=transport,
            ),
        )
        assert isinstance(result, Completed), result
        self.assertEqual(result.output, {"status": "submitted", "message_id": "outbox-1"})
        self.assertEqual(result.state["draft"], "Tuesday works for me.")
        self.assertEqual(result.state["approval"], {"type": "choice", "choice": "APPROVE"})


if __name__ == "__main__":
    unittest.main()
