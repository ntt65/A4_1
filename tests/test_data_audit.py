import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from audit_data import audit_public_exports, check_public_fields


class PublicAuditTests(unittest.TestCase):
    def test_stopword_content_matches_recorded_provenance(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "resources/data_sources.json").read_text(encoding="utf-8"))
        content = (root / "resources/stopwords_en.txt").read_text(encoding="utf-8")
        normalized = (content.rstrip() + "\n").encode("utf-8")
        self.assertEqual(hashlib.sha256(normalized).hexdigest(), manifest["stopwords"]["normalized_utf8_lf_sha256"])
        self.assertEqual(len(content.splitlines()), manifest["stopwords"]["count"])

    def test_nested_corpus_text_is_rejected(self):
        with self.assertRaises(ValueError):
            check_public_fields({"results": [{"text": "Private Example"}]}, "test")

    def test_numeric_explanations_and_public_queries_are_accepted(self):
        check_public_fields({
            "query": "space satellite",
            "results": [{"doc_id": 3, "feature_id": 42, "contribution": 0.2}],
        }, "test")

    def test_archive_cannot_publish_private_review_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            (artifacts / "run_logs").mkdir(parents=True)
            for name in ("classifier_metrics.json", "bonus_metrics.json"):
                (artifacts / name).write_text(json.dumps({"accuracy": 0.9}), encoding="utf-8")
            (artifacts / "run_logs" / "search.txt").write_text(
                "[1] Score: 0.5 | Doc #3\n", encoding="utf-8",
            )
            with zipfile.ZipFile(root / "A4_1_complete_project_source.zip", "w") as archive:
                archive.writestr("artifacts/private_review/findings.jsonl", "private review")
            with patch("audit_data.ROOT", root), self.assertRaises(ValueError):
                audit_public_exports()

    def test_stale_archive_with_document_excerpt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            (artifacts / "run_logs").mkdir(parents=True)
            names = ("classifier_metrics.json", "bonus_metrics.json")
            for name in names:
                (artifacts / name).write_text("{}", encoding="utf-8")
            (artifacts / "run_logs" / "search.txt").write_text(
                "[1] Score: 0.5 | Doc #3\n", encoding="utf-8",
            )
            with zipfile.ZipFile(root / "A4_1_complete_project_source.zip", "w") as archive:
                for name in names:
                    archive.writestr(f"artifacts/{name}", "{}")
                archive.writestr("artifacts/run_logs/search.txt", "[1] Score: 0.5 | Doc #3: Private Example")
            with patch("audit_data.ROOT", root), self.assertRaises(ValueError):
                audit_public_exports()

    def test_stale_notebook_in_archive_is_rejected(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            (artifacts / "run_logs").mkdir(parents=True)
            names = ("classifier_metrics.json", "bonus_metrics.json")
            for name in names:
                (artifacts / name).write_text("{}", encoding="utf-8")
            search_output = "[1] Score: 0.5 | Doc #3\n"
            (artifacts / "run_logs/search.txt").write_text(search_output, encoding="utf-8")
            notebook_bytes = (project / "A4_1_mission.ipynb").read_bytes()
            report_bytes = (project / "artifacts/notebook_verification.json").read_bytes()
            (root / "A4_1_mission.ipynb").write_bytes(notebook_bytes)
            (artifacts / "notebook_verification.json").write_bytes(report_bytes)
            stale = json.loads(notebook_bytes)
            stale["cells"][0]["source"] = ["# Older notebook version\n"]
            with zipfile.ZipFile(root / "A4_1_complete_project_source.zip", "w") as archive:
                for name in names:
                    archive.writestr(f"artifacts/{name}", "{}")
                archive.writestr("artifacts/run_logs/search.txt", search_output)
                archive.writestr("A4_1_mission.ipynb", json.dumps(stale))
                archive.writestr("artifacts/notebook_verification.json", report_bytes)
            with patch("audit_data.ROOT", root), self.assertRaisesRegex(ValueError, "stale notebook"):
                audit_public_exports()


if __name__ == "__main__":
    unittest.main()
