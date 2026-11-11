from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from scripts.check_commit_message import validate_message


ROOT = Path(__file__).resolve().parents[1]


class CommitMessageTests(unittest.TestCase):
    def test_conventional_subject_and_body_reference(self):
        messages = [
            "feat: implement tfidf (#2)",
            "fix(search): handle empty queries\n\nRefs #3",
            "refactor(core)!: change vectorizer API\n\nCloses #2",
            "docs: 실행 계획 갱신 (#11)",
        ]
        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(validate_message(message), [])

    def test_legacy_auto_commit_formats_are_rejected(self):
        for message in ("Edited 7 files", "docs : eval", "ai studio log"):
            with self.subTest(message=message):
                self.assertTrue(validate_message(message))

    def test_issue_reference_is_required_outside_comment_lines(self):
        self.assertTrue(validate_message("fix: correct search\n\n# Refs #3"))
        self.assertTrue(validate_message("fix: correct search (#0)"))
        self.assertTrue(validate_message("fix: correct search"))

    def test_empty_and_incomplete_subjects_are_rejected(self):
        for message in ("", "# Comment only", "fix: (#11)\n", "fix: \n\nRefs #11"):
            with self.subTest(message=message):
                # A parenthesized issue alone is not a change summary.
                self.assertTrue(validate_message(message))

    def test_native_merge_messages_are_accepted(self):
        messages = [
            "Merge pull request #12 from ntt65/chore/a4-1-git-workflow",
            "Merge branch 'feature'",
            "Merge remote-tracking branch 'origin/main'",
            "Merge tag 'v1.0'",
        ]
        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(validate_message(message), [])

    def test_validator_cli_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "COMMIT_EDITMSG"
            for message, expected in (("fix: repair issue sync (#11)", 0), ("Edited 7 files", 1)):
                path.write_text(message, encoding="utf-8")
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/check_commit_message.py"), str(path)],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, expected, result.stderr)

    @unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "A Git checkout is required for the hook integration check")
    def test_git_runs_hook_for_valid_and_invalid_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "message.txt"
            for message, expected in (("fix: repair issue sync (#11)", 0), ("Edited 7 files", 1)):
                path.write_text(message, encoding="utf-8")
                result = subprocess.run(
                    ["git", "-c", "core.hooksPath=.githooks", "hook", "run", "commit-msg", "--", str(path)],
                    cwd=ROOT, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, expected, result.stderr)


if __name__ == "__main__":
    unittest.main()
