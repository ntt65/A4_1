"""Guard the standalone notebook's implementation and saved execution evidence."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_notebook import validate_notebook, validate_saved_notebook


ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((ROOT / "A4_1_mission.ipynb").read_text(encoding="utf-8"))

    def modified_code(self):
        notebook = deepcopy(self.notebook)
        code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
        return notebook, code

    def test_saved_execution_is_complete_and_has_embedded_image(self):
        result = validate_notebook(self.notebook, require_executed=True)
        self.assertGreaterEqual(result["embedded_png_outputs"], 1)

    def test_repository_imports_are_rejected(self):
        for source in ("from src.tfidf import CustomTfidfVectorizer", "import main", "from . import tfidf"):
            with self.subTest(source=source):
                notebook, cell = self.modified_code()
                cell["source"] = source
                with self.assertRaises(ValueError):
                    validate_notebook(notebook)

    def test_error_outputs_are_rejected(self):
        notebook, cell = self.modified_code()
        cell["outputs"] = [{"output_type": "error", "ename": "ExampleError", "evalue": "failure", "traceback": []}]
        with self.assertRaisesRegex(ValueError, "error output"):
            validate_notebook(notebook)

    def test_stale_or_skipped_execution_is_rejected(self):
        notebook, cell = self.modified_code()
        cell["execution_count"] = None
        with self.assertRaisesRegex(ValueError, "execute once in order"):
            validate_notebook(notebook, require_executed=True)

    def test_private_or_active_output_is_rejected(self):
        outputs = [
            {"output_type": "stream", "name": "stdout", "text": "example@example.invalid"},
            {"output_type": "stream", "name": "stdout", "text": "C:\\Users\\example\\secret.txt"},
            {"output_type": "display_data", "metadata": {}, "data": {"text/html": "<script>example()</script>"}},
        ]
        for output in outputs:
            with self.subTest(output=output):
                notebook, cell = self.modified_code()
                cell["outputs"] = [output]
                with self.assertRaises(ValueError):
                    validate_notebook(notebook)

    def test_invalid_embedded_image_is_rejected(self):
        notebook, cell = self.modified_code()
        cell["outputs"] = [{"output_type": "display_data", "metadata": {}, "data": {"image/png": "YmFk"}}]
        with self.assertRaisesRegex(ValueError, "Invalid embedded PNG"):
            validate_notebook(notebook)

    def test_execution_report_matches_the_delivered_notebook(self):
        report = json.loads((ROOT / "artifacts/notebook_verification.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["fresh_kernel"] and report["isolated_directory"])
        self.assertFalse(report["external_repository_code"])
        self.assertEqual(report["initial_directory_files"], ["A4_1_mission.ipynb"])
        self.assertNotIn(b"\r\n", (ROOT / "A4_1_mission.ipynb").read_bytes())
        self.assertEqual(report["notebook_sha256"], hashlib.sha256((ROOT / "A4_1_mission.ipynb").read_bytes()).hexdigest())
        results = report["results"]
        for group in results["tfidf"]["groups"].values():
            self.assertLess(group["max_absolute_error"], 1e-6)
        self.assertEqual(len(results["classifier"]["misclassified_examples"]), 5)
        self.assertEqual(results["bonus"]["inverted_index"]["query_count"], 7)
        self.assertEqual(results["bonus"]["inverted_index"]["top5_rank_match_fraction"], 1.0)
        self.assertIn("pending", results["data_review_status"])

    def test_source_edited_after_execution_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "A4_1_mission.ipynb"
            changed = deepcopy(self.notebook)
            changed["cells"][0]["source"] = ["# Edited after execution\n"]
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed after its recorded execution"):
                validate_saved_notebook(path, ROOT / "artifacts/notebook_verification.json")


if __name__ == "__main__":
    unittest.main()
