"""Check the mission notebook and optionally execute it in an isolated directory."""

import argparse
import ast
from base64 import b64decode
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "A4_1_mission.ipynb"
REPO_MODULES = {"src", "main", "verify_tfidf", "audit_data", "package_project"}
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
LOCAL_PATH = re.compile(r"[A-Za-z]:[\\/]|/(?:Users|home)/")


def validate_notebook(notebook, require_executed=False):
    """Validate independent code and published outputs without optional dependencies."""
    if notebook.get("nbformat") != 4 or not notebook.get("cells"):
        raise ValueError("Expected a nonempty nbformat 4 notebook")
    ids, code_count, image_count = set(), 0, 0
    code_text = []
    for index, cell in enumerate(notebook["cells"]):
        cell_id = cell.get("id")
        if not cell_id or cell_id in ids:
            raise ValueError("Cells must have unique stable IDs")
        ids.add(cell_id)
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        if "\ufffd" in source:
            raise ValueError("Notebook source contains replacement characters")
        if cell["cell_type"] != "code":
            continue
        code_count += 1
        code_text.append(source)
        tree = ast.parse(source, filename=f"cell-{index}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                if node.level:
                    raise ValueError("Relative repository imports are not standalone")
            else:
                continue
            if any(module.split(".")[0] in REPO_MODULES for module in modules):
                raise ValueError("Notebook imports repository code")
        if require_executed and cell.get("execution_count") != code_count:
            raise ValueError("Expected every code cell to execute once in order")
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                raise ValueError("Notebook contains an error output")
            data = output.get("data", {})
            if any(mime in data for mime in ("application/javascript", "text/html")):
                raise ValueError("Unexpected active or HTML output")
            texts = [output.get("text", ""), data.get("text/plain", "")]
            for content in texts:
                content = "".join(content) if isinstance(content, list) else content
                if EMAIL.search(content) or LOCAL_PATH.search(content):
                    raise ValueError("Published output contains an email or machine path")
            if "image/png" in data:
                encoded = data["image/png"]
                encoded = "".join(encoded) if isinstance(encoded, list) else encoded
                if not b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("Invalid embedded PNG output")
                image_count += 1
    implementation = "\n".join(code_text)
    for name in ("TextPreprocessor", "CustomTfidfVectorizer", "SearchEngine", "BonusBM25"):
        if not any(isinstance(node, ast.ClassDef) and node.name == name
                   for node in ast.walk(ast.parse(implementation))):
            raise ValueError(f"Missing inline implementation: {name}")
    if require_executed and image_count < 1:
        raise ValueError("The executed notebook must embed its confusion matrix")
    return {"cells": len(notebook["cells"]), "code_cells": code_count,
            "embedded_png_outputs": image_count, "public_output_check": "PASS"}


def execute_notebook(path=NOTEBOOK, timeout=900):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    validate_notebook(notebook)
    started = perf_counter()
    with TemporaryDirectory(prefix="a4-1-notebook-") as temporary:
        workspace = Path(temporary)
        run_dir = workspace / "isolated"
        run_dir.mkdir()
        shutil.copy2(path, run_dir / path.name)
        assert len(list(run_dir.iterdir())) == 1
        kernel_dir = workspace / "kernels" / "a4-1-verify"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "A4_1 verification", "language": "python",
        }), encoding="utf-8")
        manager = KernelManager(
            kernel_name="a4-1-verify",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]),
            connection_file=str(workspace / "kernel-connection.json"),
        )
        client = NotebookClient(notebook, km=manager, timeout=timeout, allow_errors=False,
                                resources={"metadata": {"path": str(run_dir)}},
                                record_timing=False)
        client.on_cell_start = lambda cell, cell_index: print(
            f"Executing cell {cell_index + 1}/{len(notebook.cells)}", flush=True
        )
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["A4_1_DATA_HOME"] = str(ROOT / "artifacts" / "data_cache")
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["IPYTHONDIR"] = str(workspace / "ipython")
        # Explicit cwd is also needed when supplying a pre-created KernelManager.
        try:
            client.execute(cwd=str(run_dir), env=environment)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.validate(notebook)
        check = validate_notebook(notebook, require_executed=True)
        results = json.loads((run_dir / "artifacts/notebook/notebook_results.json").read_text(encoding="utf-8"))
        if results["data_review_status"] != "full corpus manual name and signature review pending (#8)":
            raise ValueError("The remaining corpus review must be reported")
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            nbformat.write(notebook, stream)
    check.update({
        "status": "PASS", "executed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "fresh_kernel": True, "isolated_directory": True,
        "initial_directory_files": [path.name],
        "data_cache": "existing sklearn cache supplied via A4_1_DATA_HOME",
        "external_repository_code": False,
        "serialization_line_endings": "LF",
        "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "results": results,
    })
    destination = ROOT / "artifacts/notebook_verification.json"
    destination.write_text(json.dumps(check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return check


def validate_saved_notebook(path=NOTEBOOK, report_path=None):
    path = Path(path)
    report_path = Path(report_path) if report_path is not None else path.parent / "artifacts/notebook_verification.json"
    result = validate_notebook(json.loads(path.read_text(encoding="utf-8")), require_executed=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if report.get("notebook_sha256") != digest or report.get("status") != "PASS":
        raise ValueError("Notebook changed after its recorded execution; run with --execute")
    result["execution_report_hash"] = "PASS"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run every cell in a fresh kernel outside the repository")
    parser.add_argument("--timeout", type=int, default=900, help="Per-cell timeout in seconds")
    args = parser.parse_args()
    if args.execute:
        result = execute_notebook(timeout=args.timeout)
    else:
        result = validate_saved_notebook()
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
