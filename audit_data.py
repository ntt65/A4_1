"""Inspect the local corpus and public exports without publishing review excerpts."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

from src.dataset import ROOT, load_data
from scripts.verify_notebook import validate_notebook


FORBIDDEN_FIELDS = {"text", "snippet", "term"}
PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "phone_candidate": re.compile(r"(?<!\w)(?:\+\d{1,3}[- .]?)?\(?\d{3}\)?[- .]\d{3}[- .]\d{4}(?!\w)"),
    "name_or_title_candidate": re.compile(r"\b[A-Z][a-z]{1,24}(?:[ -][A-Z][a-z]{1,24}){1,3}\b"),
}


def check_public_fields(value, location):
    if isinstance(value, dict):
        forbidden = FORBIDDEN_FIELDS.intersection(value)
        if forbidden:
            raise ValueError(f"Public export contains forbidden fields at {location}: {sorted(forbidden)}")
        for key, item in value.items():
            check_public_fields(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_public_fields(item, f"{location}[{index}]")


def check_search_output(content):
    for line in content.splitlines():
        if line.startswith("[") and not re.fullmatch(r"\[\d+\] Score: [\d.]+ \| Doc #\d+", line):
            raise ValueError("Published search output contains document text")


def audit_public_exports(check_archive=True):
    paths = [ROOT / "artifacts" / name for name in ("classifier_metrics.json", "bonus_metrics.json")]
    for path in paths:
        check_public_fields(json.loads(path.read_text(encoding="utf-8")), path.name)
    search_log = ROOT / "artifacts" / "run_logs" / "search.txt"
    check_search_output(search_log.read_text(encoding="utf-8"))
    result = {"status": "PASS", "checked_metrics": len(paths), "search_log": "numeric output", "archive": "not_checked"}
    notebook_path = ROOT / "A4_1_mission.ipynb"
    notebook_hash = None
    if notebook_path.exists():
        validate_notebook(json.loads(notebook_path.read_text(encoding="utf-8")), require_executed=True)
        notebook_hash = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
        notebook_report = json.loads((ROOT / "artifacts/notebook_verification.json").read_text(encoding="utf-8"))
        if notebook_report["notebook_sha256"] != notebook_hash:
            raise ValueError("Notebook execution report is stale")
        check_public_fields(notebook_report["results"], "notebook results")
        result["notebook"] = "PASS"
    if not check_archive:
        return result
    archive_path = ROOT / "A4_1_complete_project_source.zip"
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Source archive integrity check failed")
        for name in archive.namelist():
            if "data_cache/" in name or "private_review/" in name or name.endswith("index.pkl"):
                raise ValueError(f"Private or cached content in source archive: {name}")
        for path in paths:
            name = path.relative_to(ROOT).as_posix()
            check_public_fields(json.loads(archive.read(name)), f"ZIP/{name}")
        check_search_output(archive.read("artifacts/run_logs/search.txt").decode("utf-8"))
        if notebook_hash is not None:
            members = set(archive.namelist())
            if not {"A4_1_mission.ipynb", "artifacts/notebook_verification.json"} <= members:
                raise ValueError("Source archive is missing the verified notebook")
            packed = archive.read("A4_1_mission.ipynb")
            validate_notebook(json.loads(packed), require_executed=True)
            if hashlib.sha256(packed).hexdigest() != notebook_hash:
                raise ValueError("Source archive contains a stale notebook")
            packed_report = json.loads(archive.read("artifacts/notebook_verification.json"))
            if packed_report["notebook_sha256"] != notebook_hash:
                raise ValueError("Source archive contains a stale notebook execution report")
            check_public_fields(packed_report["results"], "ZIP/notebook results")
    result["archive"] = "PASS"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-public", action="store_true", help="Check published exports without loading the corpus")
    args = parser.parse_args()
    public_result = audit_public_exports()
    if args.check_public:
        print(json.dumps(public_result, indent=2))
        return

    train, test, _ = load_data()
    findings = []
    counts = Counter()
    documents_with_candidates = set()
    for split, documents in (("train", train[0]), ("test", test[0])):
        for doc_id, text in enumerate(documents):
            for kind, pattern in PATTERNS.items():
                for match in pattern.finditer(text):
                    counts[kind] += 1
                    documents_with_candidates.add((split, doc_id))
                    findings.append({
                        "split": split, "doc_id": doc_id, "kind": kind,
                        "match": match.group(),
                        "context": text[max(0, match.start() - 60):match.end() + 60],
                    })
    private_path = ROOT / "artifacts" / "private_review" / "findings.jsonl"
    private_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = private_path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in findings), encoding="utf-8")
    temporary.replace(private_path)

    report = {
        "checked_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "scope": "cleaned training and test documents used by the pipeline",
        "documents_checked": len(train[0]) + len(test[0]),
        "pattern_matches": {kind: counts[kind] for kind in PATTERNS},
        "documents_with_candidates": len(documents_with_candidates),
        "review_status": "manual_review_pending",
        "limitations": "Capitalized word sequences include titles and organizations. Lowercase names and other identifying information may be missed.",
        "private_findings": "artifacts/private_review/findings.jsonl",
        "public_exports": public_result,
        "follow_up_issue": "https://github.com/ntt65/A4_1/issues/8",
    }
    destination = ROOT / "artifacts" / "privacy_audit.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
