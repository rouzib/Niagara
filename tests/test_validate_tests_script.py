"""Checks for the documented direct test-validation entry point."""

import os
import subprocess
import sys
from pathlib import Path


def test_validation_entry_point_finds_project_examples_from_another_directory(tmp_path):
    project = Path(__file__).resolve().parents[1]
    script = project / "scripts" / "validate_tests.py"
    shadow = tmp_path / "examples"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("", encoding="utf-8")
    code = (
        "import importlib, pathlib, runpy, sys; "
        "runpy.run_path(sys.argv[1], run_name='validation_test'); "
        "examples = importlib.import_module('examples'); "
        "assert pathlib.Path(examples.__file__).resolve() == "
        "pathlib.Path(sys.argv[2], 'examples', '__init__.py').resolve()"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [sys.executable, "-c", code, str(script), str(project)],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
