"""Validate the repository's Conventional Commit and issue-reference format."""

import argparse
from pathlib import Path
import re
import sys


CONVENTIONAL_SUBJECT = re.compile(
    r"^(feat|fix|docs|test|refactor|chore|build|ci|perf|style|revert)"
    r"(?:\([^()\r\n]+\))?!?: (?P<summary>\S.*)$"
)
ISSUE_REFERENCE = re.compile(r"#[1-9]\d*\b")
MERGE_SUBJECT = re.compile(
    r"^Merge (?:pull request #[1-9]\d*\b|branch |remote-tracking branch |tag )"
)


def validate_message(message):
    lines = [line for line in message.splitlines() if not line.lstrip().startswith("#")]
    effective = "\n".join(lines).strip()
    if not effective:
        return ["The commit message is empty."]
    subject = effective.splitlines()[0]
    if MERGE_SUBJECT.match(subject):
        return []
    errors = []
    conventional = CONVENTIONAL_SUBJECT.fullmatch(subject)
    if not conventional:
        errors.append("Use type(scope): summary, for example fix(search): handle empty queries (#3).")
    elif not ISSUE_REFERENCE.sub("", conventional["summary"]).strip(" ()[]"):
        errors.append("Describe the change before its issue reference.")
    if not ISSUE_REFERENCE.search(effective):
        errors.append("Reference a GitHub issue in the subject or body, for example Refs #11.")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message_file", type=Path)
    args = parser.parse_args(argv)
    try:
        message = args.message_file.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        print(f"Cannot read commit message: {error}", file=sys.stderr)
        return 1
    errors = validate_message(message)
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
