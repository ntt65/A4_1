"""Build the source package from reviewed source and public measurement files."""

from pathlib import Path
import zipfile

from audit_data import audit_public_exports


ROOT = Path(__file__).resolve().parent


def build_package():
    audit_public_exports(check_archive=False)
    files = [ROOT / name for name in (
        ".gitignore", "AGENTS.md", "README.md", "requirements.txt",
        "requirements-notebook.txt", "A4_1_mission.ipynb",
        "main.py", "verify_tfidf.py", "audit_data.py", "package_project.py",
        "create_github_issues.py", ".gitattributes", ".githooks/commit-msg",
        "IMPLEMENTATION_PLAN.md", "docs/A4_1_mission.pdf", "docs/A4_1_eval.pdf",
        "docs/A4_1_mission_QnA.md", "docs/A4_1_presentation_outline.md",
        "docs/A4_1_data_sources.md", "docs/A4_1_data_review_plan.md",
        "docs/GIT_WORKFLOW.md", "docs/git_history_map.md",
        "docs/git_workflow_repair_plan.md", "docs/github_issues.json",
        "docs/A4_1_notebook_plan.md",
    )]
    for directory, pattern in (
        ("src", "**/*.py"), ("tests", "*.py"), ("scripts", "*.py"),
        ("resources", "*.txt"), ("resources", "*.json"),
        ("docs/repo_readme", "*.md"),
    ):
        files.extend(sorted((ROOT / directory).glob(pattern)))
    files.append(ROOT / "artifacts" / "confusion_matrix.png")
    files.extend(ROOT / "artifacts" / name for name in (
        "bonus_metrics.json", "classifier_metrics.json", "dataset_metadata.json",
        "tfidf_verification.json", "verification_results.json", "privacy_audit.json",
        "notebook_verification.json",
    ))
    files.extend(ROOT / "artifacts" / "run_logs" / name for name in (
        "verify_tfidf.txt", "search.txt", "classifier.txt", "bonus.txt",
    ))
    destination = ROOT / "A4_1_complete_project_source.zip"
    temporary = destination.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(ROOT).as_posix())
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("Source package integrity check failed")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    audit_public_exports()
    print(f"Source package verified: {destination.name}, {len(files)} files")
    return destination


if __name__ == "__main__":
    build_package()
