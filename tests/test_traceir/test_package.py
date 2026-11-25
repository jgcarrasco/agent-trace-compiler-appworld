"""Loading a package: what a host needs to run one in-process.

The CLI is not the only caller. A host that drives the engine itself reads a
package through these two functions rather than through the command line, so
this is where the loading contract is tested.
"""

from pathlib import Path
import tempfile
import unittest

from traceir.engine.package import load_operations, load_workflow

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "workflows/email_reply"


class LoadWorkflowTests(unittest.TestCase):
    def test_a_directory_gives_the_graph_and_the_package_it_came_from(self):
        graph, directory = load_workflow(WORKFLOW)
        self.assertEqual(directory, WORKFLOW)
        self.assertIn("nodes", graph)

    def test_a_graph_file_gives_its_own_directory(self):
        _, directory = load_workflow(WORKFLOW / "graph.json")
        self.assertEqual(directory, WORKFLOW)

    def test_a_missing_graph_names_where_it_looked(self):
        with self.assertRaisesRegex(ValueError, "no graph found at /nonexistent"):
            load_workflow(Path("/nonexistent"))


class LoadOperationsTests(unittest.TestCase):
    def test_a_package_without_code_has_no_operations_and_no_hooks(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_operations(Path(directory)), ({}, None, None))

    def test_a_package_without_the_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "operations.py").write_text("VALUE = 1\n")
            with self.assertRaisesRegex(ValueError, "WORKFLOW_OPERATIONS"):
                load_operations(Path(directory))

    def test_operations_hooks_and_sibling_modules_are_loaded(self):
        """A package may keep code beside `operations.py` and import it."""
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "beside_operations.py").write_text("def answer():\n    return 42\n")
            (package / "operations.py").write_text(
                "from beside_operations import answer\n"
                "WORKFLOW_OPERATIONS = {'fetch': lambda: answer()}\n"
                "def setup(initial_input):\n"
                "    pass\n"
                "def teardown():\n"
                "    pass\n"
            )
            operations, setup, teardown = load_operations(package)
            self.assertEqual(operations["fetch"](), 42)
            self.assertIsNotNone(setup)
            self.assertIsNotNone(teardown)


if __name__ == "__main__":
    unittest.main()
