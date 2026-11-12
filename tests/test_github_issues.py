from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import create_github_issues as issue_sync


class FakeGitHub:
    def __init__(self, issues=()):
        self.issues = {issue["number"]: deepcopy(issue) for issue in issues}
        self.next_number = max(self.issues, default=19) + 1
        self.calls = []

    def request(self, method, url, token, payload=None):
        self.calls.append((method, url, deepcopy(payload)))
        if method == "GET":
            return deepcopy(list(self.issues.values()))
        if method == "POST":
            number = self.next_number
            self.next_number += 1
            issue = {"number": number, "state": "open", **payload}
            self.issues[number] = issue
            return deepcopy(issue)
        number = int(url.rsplit("/", 1)[1])
        self.issues[number].update(payload)
        return deepcopy(self.issues[number])


class IssueSyncTests(unittest.TestCase):
    def setUp(self):
        self.catalog = issue_sync.load_catalog()

    def test_import_has_no_token_requirement_output_or_network(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen") as request:
            with redirect_stdout(output), redirect_stderr(output):
                namespace = runpy.run_path(str(Path(issue_sync.__file__)), run_name="issue_sync_import")
        self.assertTrue(callable(namespace["main"]))
        request.assert_not_called()
        self.assertEqual(output.getvalue(), "")

    def test_default_preview_is_offline_and_has_correct_phase_numbers(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen") as request:
            with redirect_stdout(output):
                result = issue_sync.main([])
        self.assertEqual(result, 0)
        request.assert_not_called()
        self.assertIn("M5 / original #5 / [M5]", output.getvalue())
        self.assertIn("M6 / original #6 / [M6]", output.getvalue())
        self.assertIn("README", self.catalog["issues"][4]["title"])
        self.assertIn("BM25", self.catalog["issues"][5]["title"])

    def test_apply_requires_a_token_before_network(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen") as request:
            with redirect_stderr(output), self.assertRaises(SystemExit) as error:
                issue_sync.main(["--apply"])
        self.assertEqual(error.exception.code, 2)
        request.assert_not_called()

    def test_environment_token_precedes_legacy_argument_and_is_not_printed(self):
        output = io.StringIO()
        summary = {"created": 0, "updated": 0, "unchanged": 6}
        with patch.dict(os.environ, {"GITHUB_TOKEN": "environment-secret"}, clear=True):
            with patch.object(issue_sync, "sync_issues", return_value=summary) as sync:
                with redirect_stdout(output):
                    self.assertEqual(issue_sync.main(["legacy-secret", "--apply"]), 0)
        self.assertEqual(sync.call_args.args[1], "environment-secret")
        self.assertNotIn("environment-secret", output.getvalue())
        self.assertNotIn("legacy-secret", output.getvalue())

    def test_listing_paginates_all_states_and_excludes_pull_requests(self):
        first = [{"number": index + 1} for index in range(99)]
        first.append({"number": 100, "pull_request": {}})
        with patch.object(issue_sync, "request_json", side_effect=[first, [{"number": 101}]]) as request:
            result = issue_sync.list_existing_issues("secret")
        self.assertEqual(len(result), 100)
        self.assertNotIn(100, [row["number"] for row in result])
        self.assertIn("state=all", request.call_args_list[0].args[1])
        self.assertTrue(request.call_args_list[1].args[1].endswith("page=2"))

    def test_existing_number_mapping_updates_docs_five_and_bonus_six(self):
        existing = [
            {"number": item["issue_number"], "title": f"[Issue #{item['issue_number']}] old title",
             "body": "old body", "state": "open", "labels": ["existing-label"]}
            for item in self.catalog["issues"]
        ]
        server = FakeGitHub(existing)
        with patch.object(issue_sync, "request_json", side_effect=server.request), redirect_stdout(io.StringIO()):
            summary = issue_sync.sync_issues(self.catalog, "secret")
        self.assertEqual(summary, {"created": 0, "updated": 6, "unchanged": 0})
        self.assertTrue(server.issues[5]["title"].startswith("[M5]"))
        self.assertTrue(server.issues[6]["title"].startswith("[M6]"))
        self.assertEqual(server.issues[5]["labels"], ["existing-label"])
        for method, _, payload in server.calls:
            if method == "PATCH":
                self.assertEqual(set(payload), {"title", "body"})

    def test_repeated_sync_uses_markers_and_skips_unchanged_issues(self):
        server = FakeGitHub()
        with patch.object(issue_sync, "request_json", side_effect=server.request), redirect_stdout(io.StringIO()):
            first = issue_sync.sync_issues(self.catalog, "secret")
            second = issue_sync.sync_issues(self.catalog, "secret")
        self.assertEqual(first["created"], 6)
        self.assertEqual(second, {"created": 0, "updated": 0, "unchanged": 6})
        self.assertEqual(sum(method == "POST" for method, _, _ in server.calls), 6)
        self.assertEqual(len(server.issues), 6)

    def test_retry_after_a_lost_create_response_does_not_duplicate(self):
        server = FakeGitHub()
        lost_response = True

        def request(method, url, token, payload=None):
            nonlocal lost_response
            result = server.request(method, url, token, payload)
            if method == "POST" and lost_response:
                lost_response = False
                raise RuntimeError("Connection lost after creation")
            return result

        with patch.object(issue_sync, "request_json", side_effect=request), redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                issue_sync.sync_issues(self.catalog, "secret")
            summary = issue_sync.sync_issues(self.catalog, "secret")
        self.assertEqual(summary, {"created": 5, "updated": 0, "unchanged": 1})
        self.assertEqual(len(server.issues), 6)

    def test_duplicate_markers_are_reported_before_writing(self):
        item = self.catalog["issues"][0]
        body = issue_sync.issue_payload(item)["body"]
        server = FakeGitHub([{"number": 20, "title": "first", "body": body},
                            {"number": 21, "title": "second", "body": body}])
        with patch.object(issue_sync, "request_json", side_effect=server.request), redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                issue_sync.sync_issues(self.catalog, "secret")
        self.assertEqual([call[0] for call in server.calls], ["GET"])

    def test_payload_normalization_keeps_one_marker_and_closed_state(self):
        item = self.catalog["issues"][0]
        payload = issue_sync.issue_payload(item)
        self.assertEqual(payload["body"].count("<!-- a4-1:M1 -->"), 1)
        server = FakeGitHub([{"number": 1, "state": "closed", **payload}])
        with patch.object(issue_sync, "request_json", side_effect=server.request), redirect_stdout(io.StringIO()):
            issue_sync.sync_issues({"issues": [item]}, "secret")
        self.assertEqual(server.issues[1]["state"], "closed")

    def test_http_error_reports_status_without_token_or_response_body(self):
        failure = urllib.error.HTTPError(issue_sync.API_URL, 403, "secret response", {}, None)
        with patch("urllib.request.urlopen", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "HTTP 403") as error:
                issue_sync.request_json("POST", issue_sync.API_URL, "private-token", {})
        self.assertNotIn("private-token", str(error.exception))
        self.assertNotIn("secret response", str(error.exception))

    def test_invalid_catalog_returns_nonzero_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "issues.json"
            path.write_text(json.dumps({"repository": issue_sync.REPO, "issues": [{"key": "M1"}]}), encoding="utf-8")
            with patch("urllib.request.urlopen") as request, redirect_stderr(io.StringIO()):
                result = issue_sync.main(["--catalog", str(path)])
        self.assertEqual(result, 1)
        request.assert_not_called()

    def test_invalid_issue_number_types_are_reported_cleanly(self):
        for number in ([], {}, True, 0, "1"):
            with self.subTest(number=number), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "issues.json"
                item = {"key": "M1", "issue_number": number, "title": "Data", "body": "Details"}
                path.write_text(json.dumps({"repository": issue_sync.REPO, "issues": [item]}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "positive integers"):
                    issue_sync.load_catalog(path)


if __name__ == "__main__":
    unittest.main()
