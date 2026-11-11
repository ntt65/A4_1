"""Preview the tracked issue catalog or explicitly synchronize it with GitHub."""

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


REPO = "ntt65/A4_1"
API_URL = f"https://api.github.com/repos/{REPO}/issues"
DEFAULT_CATALOG = Path(__file__).resolve().parent / "docs" / "github_issues.json"


def load_catalog(path=DEFAULT_CATALOG):
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(catalog, dict) or catalog.get("repository") != REPO:
        raise ValueError(f"The issue catalog must target {REPO}")
    items = catalog.get("issues", [])
    if not isinstance(items, list):
        raise ValueError("The issue catalog requires an issues list")
    keys = set()
    numbers = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each catalog entry must be an issue object")
        key = item.get("key")
        number = item.get("issue_number")
        title = item.get("title")
        body = item.get("body")
        if not all(isinstance(value, str) and value.strip() for value in (key, title, body)):
            raise ValueError("Each issue requires a nonempty key, title and body")
        if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
            raise ValueError("Original issue numbers must be positive integers")
        if key in keys or number in numbers:
            raise ValueError("Issue keys and original issue numbers must be unique")
        keys.add(key)
        numbers.add(number)
    if not items:
        raise ValueError("The issue catalog is empty")
    return catalog


def issue_payload(item):
    marker = f"<!-- a4-1:{item['key']} -->"
    body = item["body"].replace("\r\n", "\n").strip()
    if marker not in body:
        body = marker + "\n\n" + body
    return {"title": item["title"].strip(), "body": body}


def request_json(method, url, token, payload=None):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "A4_1-Issue-Sync",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        status = error.code
        error.close()
        raise RuntimeError(f"GitHub {method} failed with HTTP {status}") from None
    except urllib.error.URLError as error:
        raise RuntimeError(f"GitHub {method} connection failed") from error


def list_existing_issues(token):
    issues = []
    page = 1
    while True:
        response = request_json("GET", f"{API_URL}?state=all&per_page=100&page={page}", token)
        if not isinstance(response, list):
            raise ValueError("GitHub issue listing returned an unexpected response")
        issues.extend(item for item in response if "pull_request" not in item)
        if len(response) < 100:
            return issues
        page += 1


def find_existing(item, existing):
    marker = f"<!-- a4-1:{item['key']} -->"
    matches = [issue for issue in existing if marker in (issue.get("body") or "")]
    if not matches:
        prefix = f"[Issue #{item['issue_number']}]"
        matches = [
            issue for issue in existing
            if issue["number"] == item["issue_number"] and issue["title"].startswith(prefix)
        ]
    if not matches:
        matches = [issue for issue in existing if issue["title"] == item["title"]]
    if len(matches) > 1:
        raise ValueError(f"Multiple GitHub issues match {item['key']}; resolve the duplicate mapping")
    return matches[0] if matches else None


def sync_issues(catalog, token):
    existing = list_existing_issues(token)
    summary = {"created": 0, "updated": 0, "unchanged": 0}
    for item in catalog["issues"]:
        payload = issue_payload(item)
        current = find_existing(item, existing)
        if current is None:
            current = request_json("POST", API_URL, token, payload)
            existing.append(current)
            action = "created"
        else:
            current_body = (current.get("body") or "").replace("\r\n", "\n").strip()
            action = "unchanged"
            if current["title"] != payload["title"] or current_body != payload["body"]:
                request_json("PATCH", f"{API_URL}/{current['number']}", token, payload)
                current.update(payload)
                action = "updated"
        summary[action] += 1
        print(f"{action.upper()} #{current['number']} {payload['title']}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("token", nargs="?", help="Legacy positional token; GITHUB_TOKEN takes precedence")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--apply", action="store_true", help="Synchronize the catalog with GitHub")
    args = parser.parse_args(argv)
    try:
        catalog = load_catalog(args.catalog)
        if not args.apply:
            print(f"Catalog preview for {catalog['repository']}")
            for item in catalog["issues"]:
                print(f"{item['key']} / original #{item['issue_number']} / {item['title']}")
            print("Run with --apply to synchronize using GITHUB_TOKEN.")
            return 0
        token = os.environ.get("GITHUB_TOKEN") or args.token
        if not token:
            parser.error("--apply requires GITHUB_TOKEN or the legacy positional token")
        summary = sync_issues(catalog, token)
        print("Synchronization complete " + json.dumps(summary))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Issue synchronization failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
