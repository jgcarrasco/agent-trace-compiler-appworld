"""End-to-end tests of the command-line runner, driven by piped stdin.

No model call and no network: both roles default to the debug backend, which
reads answers from stdin, and the example's own `request_input` reads its reply
from the same stdin. Backend selection from the arguments is checked directly,
without any request.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from traceir.engine.backend import DebugBackend, DecisionBackend, GenerationBackend
from traceir.engine.cli import (
    DECIDE_KEY,
    DECIDE_MODEL,
    DECIDE_URL,
    GENERATE_KEY,
    GENERATE_MODEL,
    GENERATE_URL,
    build_backends,
    load_env_file,
)

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "workflows/email_reply"
# The example's own ask operation prints the review, so a fresh copy of it on
# the screen means the workflow asked again.
ASK = "\nTo: marta@example.com"


def run_cli(stdin: str, *arguments: str) -> subprocess.CompletedProcess:
    """Run the CLI with piped answers.

    `--env-file` points at nothing so the suite cannot be changed by whatever
    credentials happen to sit in the repository's own .env.
    """
    return subprocess.run(
        [
            sys.executable, "-m", "traceir.engine",
            "--workflow", str(WORKFLOW),
            "--env-file", "/nonexistent/.env",
            *arguments,
        ],
        cwd=ROOT, input=stdin, capture_output=True, text=True, timeout=120,
    )


def state_of(stdout: str) -> dict:
    return json.loads(stdout.split("State:\n", 1)[1])


def output_of(stdout: str) -> dict:
    return json.loads(stdout.split("Output:\n", 1)[1].split("\n\nState:")[0])


def as_generator(backend) -> GenerationBackend:
    assert isinstance(backend, GenerationBackend), backend
    return backend


def as_decider(backend) -> DecisionBackend:
    assert isinstance(backend, DecisionBackend), backend
    return backend


def as_debug(backend) -> DebugBackend:
    """The debug backend hands the engine its bound methods."""
    owner = getattr(backend, "__self__", None)
    assert isinstance(owner, DebugBackend), backend
    return owner


def argument_namespace(**overrides):
    """The fields `build_backends` reads, with debug answering both roles."""
    values = dict(
        generate_backend="debug",
        decide_backend="debug",
        env_file=Path("/nonexistent/.env"),
        max_tokens=512,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class RunnerCliTests(unittest.TestCase):
    def test_approved_run_reports_one_submitted_message(self):
        result = run_cli("Tuesday works, see you then!\n.\nYes, please send it.\n1\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Completed after 5 steps.", result.stdout)
        self.assertEqual(
            output_of(result.stdout), {"status": "submitted", "message_id": "message-1"}
        )
        state = state_of(result.stdout)
        self.assertEqual(state["draft"], "Tuesday works, see you then!")
        self.assertEqual(state["approval"], {"type": "choice", "choice": "APPROVE"})
        self.assertIn("marta@example.com", result.stdout)  # the review shown to the user

    def test_multi_line_draft_is_kept_unchanged(self):
        draft = "Hi Marta,\n\nTuesday at 15:00 works for me.\n\nSee you then!"
        result = run_cli(f"{draft}\n.\nYes, send it.\n1\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(state_of(result.stdout)["draft"], draft)

    def test_declined_run_reports_not_sent(self):
        result = run_cli("Tuesday works, see you then!\n.\nNo thanks, don't send it.\n2\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            output_of(result.stdout), {"status": "not_sent", "message_id": None}
        )

    def test_unclear_answer_asks_again_about_the_same_draft(self):
        stdin = "A draft.\n.\nMaybe later.\n3\nYes, send it.\n1\n"
        result = run_cli(stdin)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count(ASK), 2)
        self.assertEqual(output_of(result.stdout)["status"], "submitted")

    def test_unclear_loop_hits_the_step_limit(self):
        stdin = "A draft.\n.\n" + "Maybe later.\n3\n" * 20
        result = run_cli(stdin, "--max-steps", "6")
        self.assertEqual(result.returncode, 1)
        self.assertIn("step limit of 6", result.stderr)
        # What the workflow established before failing is still reported.
        self.assertEqual(state_of(result.stdout)["draft"], "A draft.")

    def test_invalid_choice_is_reported(self):
        result = run_cli("A draft.\n.\nYes, send it.\nbanana\n1\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("expected one of: APPROVE, DECLINE, UNCLEAR", result.stdout)

    def test_json_value_is_accepted_on_one_line(self):
        result = run_cli('{"subject": "Re: Tuesday"}\n\nYes.\n1\n')
        self.assertEqual(result.returncode, 1)
        self.assertIn("'draft_reply' value does not match its schema", result.stderr)

    def test_input_that_does_not_match_the_graph_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text("{}")
            result = run_cli("", "--input", str(path))
        self.assertEqual(result.returncode, 1)
        self.assertIn("Invalid input:", result.stderr)
        self.assertIn("'email' is a required property", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_missing_input_file_is_reported(self):
        result = run_cli("", "--input", "/nonexistent/input.json")
        # A workflow problem, reported like the others rather than as a usage error.
        self.assertEqual(result.returncode, 1)
        self.assertIn("no initial input", result.stderr)

    def test_unknown_workflow_path_is_reported(self):
        result = subprocess.run(
            [sys.executable, "-m", "traceir.engine", "--workflow", "/nonexistent"],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("no graph found", result.stderr)


class LifecycleHookTests(unittest.TestCase):
    """A package's own setup and teardown are called around the run."""

    def run_package(self, operations_source: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "graph.json").write_text((WORKFLOW / "graph.json").read_text())
            (package / "input.example.json").write_text((WORKFLOW / "input.example.json").read_text())
            (package / "operations.py").write_text(operations_source)
            return subprocess.run(
                [
                    sys.executable, "-m", "traceir.engine",
                    "--workflow", str(package),
                    "--env-file", "/nonexistent/.env",
                ],
                cwd=ROOT,
                input="A draft.\n.\nYes, send it.\n1\n",
                capture_output=True, text=True, timeout=120,
            )

    def test_setup_and_teardown_run_around_the_workflow(self):
        source = (WORKFLOW / "operations.py").read_text() + """
import os
def setup(initial_input):
    print(f"SETUP {initial_input['email']['id']}")
def teardown():
    print("TEARDOWN")
"""
        result = self.run_package(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SETUP email_123", result.stdout)
        self.assertIn("TEARDOWN", result.stdout)
        # The hooks run inside the engine, so both print before the report the
        # CLI writes afterwards.
        self.assertLess(result.stdout.index("SETUP"), result.stdout.index("TEARDOWN"))
        self.assertLess(result.stdout.index("TEARDOWN"), result.stdout.index("Trace:"))

    def test_a_package_without_hooks_still_runs(self):
        result = self.run_package((WORKFLOW / "operations.py").read_text())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Completed after", result.stdout)


class UsageOutTests(unittest.TestCase):
    """The flag reports the calls a run made, and is written even when it fails."""

    def run_with_usage(self, stdin: str, *arguments: str):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "usage.json"
        result = run_cli(stdin, "--usage-out", str(path), *arguments)
        return result, path

    def test_a_terminal_run_records_no_model_calls(self):
        result, path = self.run_with_usage("A draft.\n.\nYes, send it.\n1\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(path.read_text()), [])

    def test_a_failed_run_still_writes_what_it_spent(self):
        stdin = "A draft.\n.\n" + "Maybe later.\n3\n" * 20
        result, path = self.run_with_usage(stdin, "--max-steps", "6")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(path.read_text()), [])


class BackendWiringTests(unittest.TestCase):
    """The flags resolve to backends without making any request."""

    CHAT_URL = "http://localhost:8080/v1/chat/completions"
    CHAT_MODEL = "local-7b"
    DECIDE_ENDPOINT = "https://decide.test/api/alpha/decisions"
    DECIDE_MODEL_ID = "jev"

    def setUp(self):
        patcher = patch.dict(
            os.environ,
            {
                GENERATE_URL: self.CHAT_URL,
                GENERATE_MODEL: self.CHAT_MODEL,
                GENERATE_KEY: "chat-key",
                DECIDE_URL: self.DECIDE_ENDPOINT,
                DECIDE_MODEL: self.DECIDE_MODEL_ID,
                DECIDE_KEY: "decide-key",
            },
            clear=False,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def arguments(self, **overrides):
        return argument_namespace(**overrides)

    def test_debug_answers_both_roles_by_default_and_needs_no_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            generate, decide = build_backends(self.arguments())
        # Bound methods of the debug backend, dropped straight into the engine.
        self.assertIs(as_debug(generate), as_debug(decide))

    def test_each_role_builds_its_own_client_from_its_own_key(self):
        generate, decide = build_backends(
            self.arguments(generate_backend="chat", decide_backend="decisions")
        )
        generator, decider = as_generator(generate), as_decider(decide)
        self.assertEqual(
            (generator.model, generator.api_key, generator.url),
            (self.CHAT_MODEL, "chat-key", self.CHAT_URL),
        )
        self.assertEqual(
            (decider.model, decider.api_key, decider.url),
            (self.DECIDE_MODEL_ID, "decide-key", self.DECIDE_ENDPOINT),
        )

    def test_one_role_can_stay_with_the_terminal(self):
        generate, decide = build_backends(self.arguments(generate_backend="chat"))
        self.assertIsInstance(generate, GenerationBackend)
        as_debug(decide)

    def test_a_missing_variable_names_itself_and_the_file_it_looked_in(self):
        env_file = Path("/tmp/absent.env")
        with patch.dict(os.environ, {DECIDE_KEY: "k"}, clear=True):
            with self.assertRaisesRegex(
                ValueError, f"{GENERATE_URL} is not set; export it or add it to {env_file}"
            ):
                build_backends(
                    self.arguments(generate_backend="chat", env_file=env_file)
                )
        with patch.dict(
            os.environ, {GENERATE_URL: self.CHAT_URL, GENERATE_MODEL: "m"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, f"{GENERATE_KEY} is not set"):
                build_backends(self.arguments(generate_backend="chat"))


class EnvFileTests(unittest.TestCase):
    """A local .env is enough, but an exported variable still wins."""

    def test_unset_variables_are_filled_and_existing_ones_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('# comments and blanks are skipped\n\nEXPORTED="kept"\nNEW=from-file\n')
            with patch.dict(os.environ, {"EXPORTED": "from-shell"}, clear=True):
                load_env_file(path)
                self.assertEqual(os.environ["EXPORTED"], "from-shell")
                self.assertEqual(os.environ["NEW"], "from-file")

    def test_a_missing_file_is_not_an_error(self):
        with patch.dict(os.environ, {}, clear=True):
            load_env_file(Path("/nonexistent/.env"))


if __name__ == "__main__":
    unittest.main()
